/*! \file
  \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"

#include "../debug_utils.h"


namespace cutlass {
namespace gemm {
namespace warp {


/// Tile access iterator
/// Each iteration acess in the tile is used as multiplicand for one
/// warp-level matrix multiplication
template <
    /// Size of the tile (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand_,
    /// Data type of A elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Delta between *MMA operations (in units of *MMA operations, concept:
    /// MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads = 32,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1
>
class MmaTensorOpMultiplicandCongruousTileAccessIterator {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(
      kOperand == Operand::kA,
      "MmaTensorOpMultiplicandIterator may only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Number of elements accessed per Shared Memory load
  /// each access is a 32bit word.
  static constexpr int kElementsPerAccess = cutlass::const_max(
      32 / sizeof_bits<Element>::value, 1);

  using InstructionCount = MatrixShape<
    Shape::kRow / InstructionShape::kRow,
    Shape::kColumn / InstructionShape::kColumn
  >;

  /// Number of times this iterator can be incremented before it loops
  /// back to the beginning of the smem buffer
  static int const kIterations = InstructionCount::kColumn;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kRow * InstructionShape::kColumn / kThreads>;

  /// Memory access type
  using AccessType = AlignedArray<Element, kElementsPerAccess>;

private:

  /// Underlying tensor reference
  TensorRef ref_;

  /// Starting coordinates
  MatrixCoord origin_;

  /// Iterations in a tile
  int iterations_;

public:

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandCongruousTileAccessIterator(TensorRef const &ref, int lane_id)
  : ref_(ref), iterations_(0)
  {
    origin_ = MatrixCoord(
        lane_id / 4,
        (lane_id % 4) * kElementsPerAccess);

    // constexpr int m = 8;
    // constexpr int n = 16;
    // PRINT_T0_L0("tensor ref (%d, %d)", m, n);
    // print_tensor_ref(ref, m, n);
  }


  /// Advances iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandCongruousTileAccessIterator& add_tile_offset(
      TensorCoord const& tile_offset) {

    TensorCoord coord_offset(
        tile_offset.row() * Shape::kRow,
        tile_offset.column() * Shape::kColumn);

    origin_ += coord_offset;

    return *this;
  }


  /// increase iterations in a tile
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandCongruousTileAccessIterator& operator++() {
    iterations_++;

    if (iterations_ >= InstructionCount::kColumn) {
      // Advance the iterator along the advance dimension
      add_tile_offset({0, 1});  // Operand A advances along columns
      iterations_ = 0;
    }

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    AccessType *access_ptr = reinterpret_cast<AccessType *>(&frag);

    // "inner" iterations are over rows
    CUTLASS_PRAGMA_UNROLL
    for (int inst_m_idx = 0; inst_m_idx < InstructionCount::kRow; ++inst_m_idx) {
      // Take advantage of Tensor Op's 8 x 4T access pattern
      constexpr int kAccessesPerInstructionM = InstructionShape::kRow / 8;
      constexpr int kAccessesPerInstructionN = InstructionShape::kColumn / kElementsPerAccess / 4;

      // iterate over access N indices within instruction
      CUTLASS_PRAGMA_UNROLL
      for (int access_n_idx = 0; access_n_idx < kAccessesPerInstructionN; ++access_n_idx) {

        // iterate over access M indices within instruction
        CUTLASS_PRAGMA_UNROLL
        for (int access_m_idx = 0; access_m_idx < kAccessesPerInstructionM; ++access_m_idx) {
          const int access_idx =
              access_m_idx +
              kAccessesPerInstructionM * (access_n_idx + kAccessesPerInstructionN * inst_m_idx);
          MatrixCoord offset(
            access_m_idx * 8 + inst_m_idx * InstructionShape::kRow,
            access_n_idx * 4 * kElementsPerAccess + iterations_ * InstructionShape::kColumn);

          MatrixCoord total_offset = origin_ + offset;

          access_ptr[access_idx] = *reinterpret_cast<AccessType const *>(
            ref_.data() + ref_.offset(total_offset));

          // cutlass::NumericConverter<float, typename AccessType::Element> converter{};

          // auto print = [&](int warp_id, int lane_id) {
          //   PRINT_TN_LN(
          //       warp_id,
          //       lane_id,
          //       "  access: (%d, %d): (%d, %d) + (%d, %d) = (%d, %d) -> [%.3f, %.3f]",
          //       access_m_idx, access_n_idx,
          //       origin_.row(), origin_.column(),
          //       offset.row(), offset.column(),
          //       total_offset.row(), total_offset.column(),
          //       converter(access_ptr[access_idx][0]), converter(access_ptr[access_idx][1]));
          // };

          // for (int lane_id = 0; lane_id < 12; ++lane_id) {
          //   if (threadIdx.x == 0 && threadIdx.y == lane_id) {
          //     print(0, lane_id);
          //   }
          // }
        }
      }
    }
  }
};

} // namespace warp
} // namespace gemm
} // namespace cutlass
