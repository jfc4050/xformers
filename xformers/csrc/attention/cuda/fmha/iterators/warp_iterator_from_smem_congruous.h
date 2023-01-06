/*! \file
  \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/numeric_conversion.h"
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
class MmaTensorOpMultiplicandCongruousTileAccessIterator;


template <
    typename Shape_,
    typename Element_,
    typename Layout_,
    typename InstructionShape_,
    int OpDelta_,
    int Threads,
    int PartitionsK_
>
class MmaTensorOpMultiplicandCongruousTileAccessIterator<Shape_, Operand::kA, Element_, Layout_, InstructionShape_, OpDelta_, Threads, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kA;

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
        }
      }
    }
  }
};

template <
    typename Shape_,
    typename Element_,
    typename Layout_,
    typename InstructionShape_,
    int OpDelta_,
    int Threads,
    int PartitionsK_
>
class MmaTensorOpMultiplicandCongruousTileAccessIterator<Shape_, Operand::kB, Element_, Layout_, InstructionShape_, OpDelta_, Threads, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static const Operand kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static constexpr int kOpDelta = OpDelta_;

  /// Number of participating threads
  static constexpr int kThreads = 32;

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
  static int const kIterations = InstructionCount::kRow;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kColumn * InstructionShape::kRow / kThreads>;

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
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandCongruousTileAccessIterator(TensorRef const &ref, int lane_id)
  : ref_(ref), iterations_(0)
  {
    origin_ = MatrixCoord((lane_id % 4 * kElementsPerAccess), lane_id / 4);
  }

  /// Advances iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandCongruousTileAccessIterator& add_tile_offset(
      TensorCoord const& tile_offset) {

    TensorCoord coord_offset(
        tile_offset.row() * Shape::kRow,
        tile_offset.column() * Shape::kColumn);

    origin_ += coord_offset;

    return *this;
  }


  /// increase iterations in a tile
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandCongruousTileAccessIterator& operator++() {
    iterations_++;

    if (iterations_ >= InstructionCount::kRow) {
      // Advance the iterator along the advance dimension
      add_tile_offset({1, 0});  // Operand B advances along rows
      iterations_ = 0;
    }

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_DEVICE
  void load(Fragment &frag) const {
    // PRINT_T0_L0("layout: %s, access %s with %s", __get_type_name<Layout>().data, __get_type_name<Fragment>().data, __get_type_name<AccessType>().data);
    // PRINT_T0_L0(
    //   "overall shape: %s, instruction shape: %s, instruction count: %s",
    //   __get_type_name<Shape>().data,
    //   __get_type_name<InstructionShape>().data,
    //   __get_type_name<InstructionCount>().data);

    AccessType *access_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int inst_n_idx = 0; inst_n_idx < InstructionCount::kColumn; ++inst_n_idx) {
      constexpr int kAccessesInner = (InstructionShape::kRow / kElementsPerAccess) / 4;

      CUTLASS_PRAGMA_UNROLL
      for (int inner_idx = 0; inner_idx < kAccessesInner; ++inner_idx) {

        const int access_idx = inner_idx + kAccessesInner * inst_n_idx;

        // TODO. this is a pretty inefficient way of doing this
        for (int elt_idx_in_access = 0; elt_idx_in_access < kElementsPerAccess; ++elt_idx_in_access) {
          MatrixCoord offset(
            inner_idx * 4 * kElementsPerAccess + iterations_ * InstructionShape::kRow + elt_idx_in_access,
            inst_n_idx * 8);
          MatrixCoord access_coord = origin_ + offset;
          access_ptr[access_idx][elt_idx_in_access] = ref_.at(access_coord);
        }

        // access_ptr[access_idx] = *reinterpret_cast<AccessType const *>(
        //   ref_.data() + ref_.offset(access_coord));

        // PRINT_T0_L0("access coord: (%d, %d), idx: %d", access_coord.row(), access_coord.column(), access_idx);
        // for (int lane_id = 0; lane_id < 32; ++lane_id) {
        //   PRINT_TN_LN(
        //     0,
        //     lane_id,
        //     "access_idx: %d, access_coord: (%d, %d) + (%d, %d) = (%d, %d) -> %d -> [%f, %f]",
        //     access_idx,
        //     origin_.row(),
        //     origin_.column(),
        //     offset.row(),
        //     offset.column(),
        //     access_coord.row(),
        //     access_coord.column(),
        //     (int) ref_.offset(access_coord),
        //     cutlass::NumericConverter<float, typename AccessType::Element>::convert(access_ptr[access_idx][0]),
        //     cutlass::NumericConverter<float, typename AccessType::Element>::convert(access_ptr[access_idx][1]));
        //   __syncthreads();
        // }
      }
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int) {}
};

} // namespace warp
} // namespace gemm
} // namespace cutlass
