��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.12unknown8ܰ
m
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_name
Variable
f
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:	�@*
dtype0
q

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_name
Variable_1
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	�@*
dtype0
h

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
�
!lista__toeplitz_2/conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!lista__toeplitz_2/conv1d_4/kernel
�
5lista__toeplitz_2/conv1d_4/kernel/Read/ReadVariableOpReadVariableOp!lista__toeplitz_2/conv1d_4/kernel*#
_output_shapes
:�*
dtype0
�
!lista__toeplitz_2/conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!lista__toeplitz_2/conv1d_5/kernel
�
5lista__toeplitz_2/conv1d_5/kernel/Read/ReadVariableOpReadVariableOp!lista__toeplitz_2/conv1d_5/kernel*#
_output_shapes
:�*
dtype0
h

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_3
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
h

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_4
a
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
: *
dtype0
h

Variable_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_5
a
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
{
Adam/Variable/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@* 
shared_nameAdam/Variable/m
t
#Adam/Variable/m/Read/ReadVariableOpReadVariableOpAdam/Variable/m*
_output_shapes
:	�@*
dtype0

Adam/Variable/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*"
shared_nameAdam/Variable/m_1
x
%Adam/Variable/m_1/Read/ReadVariableOpReadVariableOpAdam/Variable/m_1*
_output_shapes
:	�@*
dtype0
�
(Adam/lista__toeplitz_2/conv1d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/lista__toeplitz_2/conv1d_4/kernel/m
�
<Adam/lista__toeplitz_2/conv1d_4/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/lista__toeplitz_2/conv1d_4/kernel/m*#
_output_shapes
:�*
dtype0
�
(Adam/lista__toeplitz_2/conv1d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/lista__toeplitz_2/conv1d_5/kernel/m
�
<Adam/lista__toeplitz_2/conv1d_5/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/lista__toeplitz_2/conv1d_5/kernel/m*#
_output_shapes
:�*
dtype0
v
Adam/Variable/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/m_2
o
%Adam/Variable/m_2/Read/ReadVariableOpReadVariableOpAdam/Variable/m_2*
_output_shapes
: *
dtype0
v
Adam/Variable/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/m_3
o
%Adam/Variable/m_3/Read/ReadVariableOpReadVariableOpAdam/Variable/m_3*
_output_shapes
: *
dtype0
v
Adam/Variable/m_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/m_4
o
%Adam/Variable/m_4/Read/ReadVariableOpReadVariableOpAdam/Variable/m_4*
_output_shapes
: *
dtype0
{
Adam/Variable/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@* 
shared_nameAdam/Variable/v
t
#Adam/Variable/v/Read/ReadVariableOpReadVariableOpAdam/Variable/v*
_output_shapes
:	�@*
dtype0

Adam/Variable/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*"
shared_nameAdam/Variable/v_1
x
%Adam/Variable/v_1/Read/ReadVariableOpReadVariableOpAdam/Variable/v_1*
_output_shapes
:	�@*
dtype0
�
(Adam/lista__toeplitz_2/conv1d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/lista__toeplitz_2/conv1d_4/kernel/v
�
<Adam/lista__toeplitz_2/conv1d_4/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/lista__toeplitz_2/conv1d_4/kernel/v*#
_output_shapes
:�*
dtype0
�
(Adam/lista__toeplitz_2/conv1d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/lista__toeplitz_2/conv1d_5/kernel/v
�
<Adam/lista__toeplitz_2/conv1d_5/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/lista__toeplitz_2/conv1d_5/kernel/v*#
_output_shapes
:�*
dtype0
v
Adam/Variable/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/v_2
o
%Adam/Variable/v_2/Read/ReadVariableOpReadVariableOpAdam/Variable/v_2*
_output_shapes
: *
dtype0
v
Adam/Variable/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/v_3
o
%Adam/Variable/v_3/Read/ReadVariableOpReadVariableOpAdam/Variable/v_3*
_output_shapes
: *
dtype0
v
Adam/Variable/v_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/v_4
o
%Adam/Variable/v_4/Read/ReadVariableOpReadVariableOpAdam/Variable/v_4*
_output_shapes
: *
dtype0

NoOpNoOp
� 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*� 
value� B�  B� 
�
hg_r
hg_i
We_r
We_i
lam_list
	alpha
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
=;
VARIABLE_VALUEVariableWe_r/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
Variable_1We_i/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
@>
VARIABLE_VALUE
Variable_2 alpha/.ATTRIBUTES/VARIABLE_VALUE
�

beta_1

beta_2
	decay
learning_rate
iterm3m4m5m6m7m8m9v:v;v<v=v>v?v@
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
�
	variables
non_trainable_variables

 layers
!metrics
	regularization_losses

trainable_variables
"layer_metrics
#layer_regularization_losses
 
][
VARIABLE_VALUE!lista__toeplitz_2/conv1d_4/kernel&hg_r/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
�
	variables
$non_trainable_variables
%metrics

&layers
regularization_losses
trainable_variables
'layer_metrics
(layer_regularization_losses
][
VARIABLE_VALUE!lista__toeplitz_2/conv1d_5/kernel&hg_i/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
�
	variables
)non_trainable_variables
*metrics

+layers
regularization_losses
trainable_variables
,layer_metrics
-layer_regularization_losses
EC
VARIABLE_VALUE
Variable_3%lam_list/0/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE
Variable_4%lam_list/1/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE
Variable_5%lam_list/2/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

.0
 
 
 
 
 
 
 
 
 
 
 
 
4
	/total
	0count
1	variables
2	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

/0
01

1	variables
`^
VARIABLE_VALUEAdam/Variable/m;We_r/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEAdam/Variable/m_1;We_i/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUE(Adam/lista__toeplitz_2/conv1d_4/kernel/mBhg_r/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUE(Adam/lista__toeplitz_2/conv1d_5/kernel/mBhg_i/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/m_2Alam_list/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/m_3Alam_list/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/m_4Alam_list/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEAdam/Variable/v;We_r/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEAdam/Variable/v_1;We_i/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUE(Adam/lista__toeplitz_2/conv1d_4/kernel/vBhg_r/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUE(Adam/lista__toeplitz_2/conv1d_5/kernel/vBhg_i/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/v_2Alam_list/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/v_3Alam_list/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/v_4Alam_list/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*+
_output_shapes
:���������@*
dtype0* 
shape:���������@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable
Variable_3
Variable_1!lista__toeplitz_2/conv1d_4/kernel!lista__toeplitz_2/conv1d_5/kernel
Variable_4
Variable_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_680379
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOp5lista__toeplitz_2/conv1d_4/kernel/Read/ReadVariableOp5lista__toeplitz_2/conv1d_5/kernel/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_4/Read/ReadVariableOpVariable_5/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp#Adam/Variable/m/Read/ReadVariableOp%Adam/Variable/m_1/Read/ReadVariableOp<Adam/lista__toeplitz_2/conv1d_4/kernel/m/Read/ReadVariableOp<Adam/lista__toeplitz_2/conv1d_5/kernel/m/Read/ReadVariableOp%Adam/Variable/m_2/Read/ReadVariableOp%Adam/Variable/m_3/Read/ReadVariableOp%Adam/Variable/m_4/Read/ReadVariableOp#Adam/Variable/v/Read/ReadVariableOp%Adam/Variable/v_1/Read/ReadVariableOp<Adam/lista__toeplitz_2/conv1d_4/kernel/v/Read/ReadVariableOp<Adam/lista__toeplitz_2/conv1d_5/kernel/v/Read/ReadVariableOp%Adam/Variable/v_2/Read/ReadVariableOp%Adam/Variable/v_3/Read/ReadVariableOp%Adam/Variable/v_4/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_680527
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1
Variable_2!lista__toeplitz_2/conv1d_4/kernel!lista__toeplitz_2/conv1d_5/kernel
Variable_3
Variable_4
Variable_5beta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/Variable/mAdam/Variable/m_1(Adam/lista__toeplitz_2/conv1d_4/kernel/m(Adam/lista__toeplitz_2/conv1d_5/kernel/mAdam/Variable/m_2Adam/Variable/m_3Adam/Variable/m_4Adam/Variable/vAdam/Variable/v_1(Adam/lista__toeplitz_2/conv1d_4/kernel/v(Adam/lista__toeplitz_2/conv1d_5/kernel/vAdam/Variable/v_2Adam/Variable/v_3Adam/Variable/v_4*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_680624��

�
�
D__inference_conv1d_4_layer_call_and_return_conditional_losses_680009

inputs/
+conv1d_expanddims_1_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
M__inference_lista__toeplitz_2_layer_call_and_return_conditional_losses_680330
input_1%
!tensordot_readvariableop_resource#
maximum_readvariableop_resource'
#tensordot_1_readvariableop_resource
conv1d_4_680018
conv1d_5_680042%
!maximum_2_readvariableop_resource%
!maximum_4_readvariableop_resource
identity�� conv1d_4/StatefulPartitionedCall�"conv1d_4/StatefulPartitionedCall_1�"conv1d_4/StatefulPartitionedCall_2�"conv1d_4/StatefulPartitionedCall_3� conv1d_5/StatefulPartitionedCall�"conv1d_5/StatefulPartitionedCall_1�"conv1d_5/StatefulPartitionedCall_2�"conv1d_5/StatefulPartitionedCall_3J
RealRealinput_1*+
_output_shapes
:���������@2
RealJ
ImagImaginput_1*+
_output_shapes
:���������@2
Imag�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/free_
Tensordot/ShapeShapeReal:output:0*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/axes:output:0Tensordot/free:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod_1:output:0Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeReal:output:0Tensordot/concat:output:0*
T0*+
_output_shapes
:@���������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMul Tensordot/ReadVariableOp:value:0Tensordot/Reshape:output:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/Const_2:output:0Tensordot/GatherV2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
	Tensordotu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	TransposeTensordot:output:0transpose/perm:output:0*
T0*,
_output_shapes
:����������2
	transposep
norm/mulMultranspose:y:0transpose:y:0*
T0*,
_output_shapes
:����������2

norm/mul�
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm/Sum/reduction_indices�
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2

norm/Sumh
	norm/SqrtSqrtnorm/Sum:output:0*
T0*,
_output_shapes
:����������2
	norm/Sqrt�
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2
norm/Squeezeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsnorm/Squeeze:output:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2

ExpandDimsu
Tile/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile/multiplesy
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*,
_output_shapes
:����������2
Tile�
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum/ReadVariableOp�
MaximumMaximumTile:output:0Maximum/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
Maximumx
ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpy
truedivRealDivReadVariableOp:value:0Maximum:z:0*
T0*,
_output_shapes
:����������2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xe
subSubsub/x:output:0truediv:z:0*
T0*,
_output_shapes
:����������2
sub`
mulMulsub:z:0transpose:y:0*
T0*,
_output_shapes
:����������2
mul�
Tensordot_1/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot_1/ReadVariableOpn
Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_1/axesu
Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_1/freec
Tensordot_1/ShapeShapeImag:output:0*
T0*
_output_shapes
:2
Tensordot_1/Shapex
Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_1/GatherV2/axis�
Tensordot_1/GatherV2GatherV2Tensordot_1/Shape:output:0Tensordot_1/free:output:0"Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_1/GatherV2|
Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_1/GatherV2_1/axis�
Tensordot_1/GatherV2_1GatherV2Tensordot_1/Shape:output:0Tensordot_1/axes:output:0$Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_1/GatherV2_1p
Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_1/Const�
Tensordot_1/ProdProdTensordot_1/GatherV2:output:0Tensordot_1/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_1/Prodt
Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_1/Const_1�
Tensordot_1/Prod_1ProdTensordot_1/GatherV2_1:output:0Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_1/Prod_1t
Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_1/concat/axis�
Tensordot_1/concatConcatV2Tensordot_1/axes:output:0Tensordot_1/free:output:0 Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_1/concat�
Tensordot_1/stackPackTensordot_1/Prod_1:output:0Tensordot_1/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_1/stack�
Tensordot_1/transpose	TransposeImag:output:0Tensordot_1/concat:output:0*
T0*+
_output_shapes
:@���������2
Tensordot_1/transpose�
Tensordot_1/ReshapeReshapeTensordot_1/transpose:y:0Tensordot_1/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot_1/Reshape�
Tensordot_1/MatMulMatMul"Tensordot_1/ReadVariableOp:value:0Tensordot_1/Reshape:output:0*
T0*(
_output_shapes
:����������2
Tensordot_1/MatMulu
Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot_1/Const_2x
Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_1/concat_1/axis�
Tensordot_1/concat_1ConcatV2Tensordot_1/Const_2:output:0Tensordot_1/GatherV2:output:0"Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_1/concat_1�
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/concat_1:output:0*
T0*,
_output_shapes
:����������2
Tensordot_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	TransposeTensordot_1:output:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_1x

norm_1/mulMultranspose_1:y:0transpose_1:y:0*
T0*,
_output_shapes
:����������2

norm_1/mul�
norm_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm_1/Sum/reduction_indices�

norm_1/SumSumnorm_1/mul:z:0%norm_1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2

norm_1/Sumn
norm_1/SqrtSqrtnorm_1/Sum:output:0*
T0*,
_output_shapes
:����������2
norm_1/Sqrt�
norm_1/SqueezeSqueezenorm_1/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2
norm_1/Squeezef
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim�
ExpandDims_1
ExpandDimsnorm_1/Squeeze:output:0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:����������2
ExpandDims_1y
Tile_1/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile_1/multiples�
Tile_1TileExpandDims_1:output:0Tile_1/multiples:output:0*
T0*,
_output_shapes
:����������2
Tile_1�
Maximum_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum_1/ReadVariableOp�
	Maximum_1MaximumTile_1:output:0 Maximum_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
	Maximum_1|
ReadVariableOp_1ReadVariableOpmaximum_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
	truediv_1RealDivReadVariableOp_1:value:0Maximum_1:z:0*
T0*,
_output_shapes
:����������2
	truediv_1W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
sub_1/xm
sub_1Subsub_1/x:output:0truediv_1:z:0*
T0*,
_output_shapes
:����������2
sub_1h
mul_1Mul	sub_1:z:0transpose_1:y:0*
T0*,
_output_shapes
:����������2
mul_1�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCallmul:z:0conv1d_4_680018*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_6800092"
 conv1d_4/StatefulPartitionedCall�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall	mul_1:z:0conv1d_5_680042*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_6800332"
 conv1d_5/StatefulPartitionedCall�
sub_2Sub)conv1d_4/StatefulPartitionedCall:output:0)conv1d_5/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:����������2
sub_2�
Tensordot_2/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot_2/ReadVariableOpn
Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_2/axesu
Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_2/freec
Tensordot_2/ShapeShapeReal:output:0*
T0*
_output_shapes
:2
Tensordot_2/Shapex
Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_2/GatherV2/axis�
Tensordot_2/GatherV2GatherV2Tensordot_2/Shape:output:0Tensordot_2/free:output:0"Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_2/GatherV2|
Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_2/GatherV2_1/axis�
Tensordot_2/GatherV2_1GatherV2Tensordot_2/Shape:output:0Tensordot_2/axes:output:0$Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_2/GatherV2_1p
Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_2/Const�
Tensordot_2/ProdProdTensordot_2/GatherV2:output:0Tensordot_2/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_2/Prodt
Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_2/Const_1�
Tensordot_2/Prod_1ProdTensordot_2/GatherV2_1:output:0Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_2/Prod_1t
Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_2/concat/axis�
Tensordot_2/concatConcatV2Tensordot_2/axes:output:0Tensordot_2/free:output:0 Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_2/concat�
Tensordot_2/stackPackTensordot_2/Prod_1:output:0Tensordot_2/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_2/stack�
Tensordot_2/transpose	TransposeReal:output:0Tensordot_2/concat:output:0*
T0*+
_output_shapes
:@���������2
Tensordot_2/transpose�
Tensordot_2/ReshapeReshapeTensordot_2/transpose:y:0Tensordot_2/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot_2/Reshape�
Tensordot_2/MatMulMatMul"Tensordot_2/ReadVariableOp:value:0Tensordot_2/Reshape:output:0*
T0*(
_output_shapes
:����������2
Tensordot_2/MatMulu
Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot_2/Const_2x
Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_2/concat_1/axis�
Tensordot_2/concat_1ConcatV2Tensordot_2/Const_2:output:0Tensordot_2/GatherV2:output:0"Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_2/concat_1�
Tensordot_2ReshapeTensordot_2/MatMul:product:0Tensordot_2/concat_1:output:0*
T0*,
_output_shapes
:����������2
Tensordot_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm�
transpose_2	TransposeTensordot_2:output:0transpose_2/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_2f
addAddV2	sub_2:z:0transpose_2:y:0*
T0*,
_output_shapes
:����������2
add�
Tensordot_3/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot_3/ReadVariableOpn
Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_3/axesu
Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_3/freec
Tensordot_3/ShapeShapeImag:output:0*
T0*
_output_shapes
:2
Tensordot_3/Shapex
Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_3/GatherV2/axis�
Tensordot_3/GatherV2GatherV2Tensordot_3/Shape:output:0Tensordot_3/free:output:0"Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_3/GatherV2|
Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_3/GatherV2_1/axis�
Tensordot_3/GatherV2_1GatherV2Tensordot_3/Shape:output:0Tensordot_3/axes:output:0$Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_3/GatherV2_1p
Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_3/Const�
Tensordot_3/ProdProdTensordot_3/GatherV2:output:0Tensordot_3/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_3/Prodt
Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_3/Const_1�
Tensordot_3/Prod_1ProdTensordot_3/GatherV2_1:output:0Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_3/Prod_1t
Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_3/concat/axis�
Tensordot_3/concatConcatV2Tensordot_3/axes:output:0Tensordot_3/free:output:0 Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_3/concat�
Tensordot_3/stackPackTensordot_3/Prod_1:output:0Tensordot_3/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_3/stack�
Tensordot_3/transpose	TransposeImag:output:0Tensordot_3/concat:output:0*
T0*+
_output_shapes
:@���������2
Tensordot_3/transpose�
Tensordot_3/ReshapeReshapeTensordot_3/transpose:y:0Tensordot_3/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot_3/Reshape�
Tensordot_3/MatMulMatMul"Tensordot_3/ReadVariableOp:value:0Tensordot_3/Reshape:output:0*
T0*(
_output_shapes
:����������2
Tensordot_3/MatMulu
Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot_3/Const_2x
Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_3/concat_1/axis�
Tensordot_3/concat_1ConcatV2Tensordot_3/Const_2:output:0Tensordot_3/GatherV2:output:0"Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_3/concat_1�
Tensordot_3ReshapeTensordot_3/MatMul:product:0Tensordot_3/concat_1:output:0*
T0*,
_output_shapes
:����������2
Tensordot_3y
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_3/perm�
transpose_3	TransposeTensordot_3:output:0transpose_3/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_3f
sub_3Subadd:z:0transpose_3:y:0*
T0*,
_output_shapes
:����������2
sub_3l

norm_2/mulMul	sub_3:z:0	sub_3:z:0*
T0*,
_output_shapes
:����������2

norm_2/mul�
norm_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm_2/Sum/reduction_indices�

norm_2/SumSumnorm_2/mul:z:0%norm_2/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2

norm_2/Sumn
norm_2/SqrtSqrtnorm_2/Sum:output:0*
T0*,
_output_shapes
:����������2
norm_2/Sqrt�
norm_2/SqueezeSqueezenorm_2/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2
norm_2/Squeezef
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_2/dim�
ExpandDims_2
ExpandDimsnorm_2/Squeeze:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:����������2
ExpandDims_2y
Tile_2/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile_2/multiples�
Tile_2TileExpandDims_2:output:0Tile_2/multiples:output:0*
T0*,
_output_shapes
:����������2
Tile_2�
Maximum_2/ReadVariableOpReadVariableOp!maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum_2/ReadVariableOp�
	Maximum_2MaximumTile_2:output:0 Maximum_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
	Maximum_2~
ReadVariableOp_2ReadVariableOp!maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2�
	truediv_2RealDivReadVariableOp_2:value:0Maximum_2:z:0*
T0*,
_output_shapes
:����������2
	truediv_2W
sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
sub_4/xm
sub_4Subsub_4/x:output:0truediv_2:z:0*
T0*,
_output_shapes
:����������2
sub_4b
mul_2Mul	sub_4:z:0	sub_3:z:0*
T0*,
_output_shapes
:����������2
mul_2�
"conv1d_4/StatefulPartitionedCall_1StatefulPartitionedCall	mul_1:z:0conv1d_4_680018*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_6800092$
"conv1d_4/StatefulPartitionedCall_1�
"conv1d_5/StatefulPartitionedCall_1StatefulPartitionedCall	mul_2:z:0conv1d_5_680042*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_6800332$
"conv1d_5/StatefulPartitionedCall_1�
add_1AddV2+conv1d_4/StatefulPartitionedCall_1:output:0+conv1d_5/StatefulPartitionedCall_1:output:0*
T0*,
_output_shapes
:����������2
add_1�
Tensordot_4/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot_4/ReadVariableOpn
Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_4/axesu
Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_4/freec
Tensordot_4/ShapeShapeImag:output:0*
T0*
_output_shapes
:2
Tensordot_4/Shapex
Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_4/GatherV2/axis�
Tensordot_4/GatherV2GatherV2Tensordot_4/Shape:output:0Tensordot_4/free:output:0"Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_4/GatherV2|
Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_4/GatherV2_1/axis�
Tensordot_4/GatherV2_1GatherV2Tensordot_4/Shape:output:0Tensordot_4/axes:output:0$Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_4/GatherV2_1p
Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_4/Const�
Tensordot_4/ProdProdTensordot_4/GatherV2:output:0Tensordot_4/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_4/Prodt
Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_4/Const_1�
Tensordot_4/Prod_1ProdTensordot_4/GatherV2_1:output:0Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_4/Prod_1t
Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_4/concat/axis�
Tensordot_4/concatConcatV2Tensordot_4/axes:output:0Tensordot_4/free:output:0 Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_4/concat�
Tensordot_4/stackPackTensordot_4/Prod_1:output:0Tensordot_4/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_4/stack�
Tensordot_4/transpose	TransposeImag:output:0Tensordot_4/concat:output:0*
T0*+
_output_shapes
:@���������2
Tensordot_4/transpose�
Tensordot_4/ReshapeReshapeTensordot_4/transpose:y:0Tensordot_4/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot_4/Reshape�
Tensordot_4/MatMulMatMul"Tensordot_4/ReadVariableOp:value:0Tensordot_4/Reshape:output:0*
T0*(
_output_shapes
:����������2
Tensordot_4/MatMulu
Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot_4/Const_2x
Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_4/concat_1/axis�
Tensordot_4/concat_1ConcatV2Tensordot_4/Const_2:output:0Tensordot_4/GatherV2:output:0"Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_4/concat_1�
Tensordot_4ReshapeTensordot_4/MatMul:product:0Tensordot_4/concat_1:output:0*
T0*,
_output_shapes
:����������2
Tensordot_4y
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_4/perm�
transpose_4	TransposeTensordot_4:output:0transpose_4/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_4j
add_2AddV2	add_1:z:0transpose_4:y:0*
T0*,
_output_shapes
:����������2
add_2�
Tensordot_5/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot_5/ReadVariableOpn
Tensordot_5/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_5/axesu
Tensordot_5/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_5/freec
Tensordot_5/ShapeShapeReal:output:0*
T0*
_output_shapes
:2
Tensordot_5/Shapex
Tensordot_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_5/GatherV2/axis�
Tensordot_5/GatherV2GatherV2Tensordot_5/Shape:output:0Tensordot_5/free:output:0"Tensordot_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_5/GatherV2|
Tensordot_5/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_5/GatherV2_1/axis�
Tensordot_5/GatherV2_1GatherV2Tensordot_5/Shape:output:0Tensordot_5/axes:output:0$Tensordot_5/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_5/GatherV2_1p
Tensordot_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_5/Const�
Tensordot_5/ProdProdTensordot_5/GatherV2:output:0Tensordot_5/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_5/Prodt
Tensordot_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_5/Const_1�
Tensordot_5/Prod_1ProdTensordot_5/GatherV2_1:output:0Tensordot_5/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_5/Prod_1t
Tensordot_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_5/concat/axis�
Tensordot_5/concatConcatV2Tensordot_5/axes:output:0Tensordot_5/free:output:0 Tensordot_5/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_5/concat�
Tensordot_5/stackPackTensordot_5/Prod_1:output:0Tensordot_5/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_5/stack�
Tensordot_5/transpose	TransposeReal:output:0Tensordot_5/concat:output:0*
T0*+
_output_shapes
:@���������2
Tensordot_5/transpose�
Tensordot_5/ReshapeReshapeTensordot_5/transpose:y:0Tensordot_5/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot_5/Reshape�
Tensordot_5/MatMulMatMul"Tensordot_5/ReadVariableOp:value:0Tensordot_5/Reshape:output:0*
T0*(
_output_shapes
:����������2
Tensordot_5/MatMulu
Tensordot_5/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot_5/Const_2x
Tensordot_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_5/concat_1/axis�
Tensordot_5/concat_1ConcatV2Tensordot_5/Const_2:output:0Tensordot_5/GatherV2:output:0"Tensordot_5/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_5/concat_1�
Tensordot_5ReshapeTensordot_5/MatMul:product:0Tensordot_5/concat_1:output:0*
T0*,
_output_shapes
:����������2
Tensordot_5y
transpose_5/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_5/perm�
transpose_5	TransposeTensordot_5:output:0transpose_5/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_5j
add_3AddV2	add_2:z:0transpose_5:y:0*
T0*,
_output_shapes
:����������2
add_3l

norm_3/mulMul	add_3:z:0	add_3:z:0*
T0*,
_output_shapes
:����������2

norm_3/mul�
norm_3/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm_3/Sum/reduction_indices�

norm_3/SumSumnorm_3/mul:z:0%norm_3/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2

norm_3/Sumn
norm_3/SqrtSqrtnorm_3/Sum:output:0*
T0*,
_output_shapes
:����������2
norm_3/Sqrt�
norm_3/SqueezeSqueezenorm_3/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2
norm_3/Squeezef
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_3/dim�
ExpandDims_3
ExpandDimsnorm_3/Squeeze:output:0ExpandDims_3/dim:output:0*
T0*,
_output_shapes
:����������2
ExpandDims_3y
Tile_3/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile_3/multiples�
Tile_3TileExpandDims_3:output:0Tile_3/multiples:output:0*
T0*,
_output_shapes
:����������2
Tile_3�
Maximum_3/ReadVariableOpReadVariableOp!maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum_3/ReadVariableOp�
	Maximum_3MaximumTile_3:output:0 Maximum_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
	Maximum_3~
ReadVariableOp_3ReadVariableOp!maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_3�
	truediv_3RealDivReadVariableOp_3:value:0Maximum_3:z:0*
T0*,
_output_shapes
:����������2
	truediv_3W
sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
sub_5/xm
sub_5Subsub_5/x:output:0truediv_3:z:0*
T0*,
_output_shapes
:����������2
sub_5b
mul_3Mul	sub_5:z:0	add_3:z:0*
T0*,
_output_shapes
:����������2
mul_3�
"conv1d_4/StatefulPartitionedCall_2StatefulPartitionedCall	mul_2:z:0conv1d_4_680018*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_6800092$
"conv1d_4/StatefulPartitionedCall_2�
"conv1d_5/StatefulPartitionedCall_2StatefulPartitionedCall	mul_3:z:0conv1d_5_680042*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_6800332$
"conv1d_5/StatefulPartitionedCall_2�
sub_6Sub+conv1d_4/StatefulPartitionedCall_2:output:0+conv1d_5/StatefulPartitionedCall_2:output:0*
T0*,
_output_shapes
:����������2
sub_6�
Tensordot_6/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot_6/ReadVariableOpn
Tensordot_6/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_6/axesu
Tensordot_6/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_6/freec
Tensordot_6/ShapeShapeReal:output:0*
T0*
_output_shapes
:2
Tensordot_6/Shapex
Tensordot_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_6/GatherV2/axis�
Tensordot_6/GatherV2GatherV2Tensordot_6/Shape:output:0Tensordot_6/free:output:0"Tensordot_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_6/GatherV2|
Tensordot_6/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_6/GatherV2_1/axis�
Tensordot_6/GatherV2_1GatherV2Tensordot_6/Shape:output:0Tensordot_6/axes:output:0$Tensordot_6/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_6/GatherV2_1p
Tensordot_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_6/Const�
Tensordot_6/ProdProdTensordot_6/GatherV2:output:0Tensordot_6/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_6/Prodt
Tensordot_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_6/Const_1�
Tensordot_6/Prod_1ProdTensordot_6/GatherV2_1:output:0Tensordot_6/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_6/Prod_1t
Tensordot_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_6/concat/axis�
Tensordot_6/concatConcatV2Tensordot_6/axes:output:0Tensordot_6/free:output:0 Tensordot_6/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_6/concat�
Tensordot_6/stackPackTensordot_6/Prod_1:output:0Tensordot_6/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_6/stack�
Tensordot_6/transpose	TransposeReal:output:0Tensordot_6/concat:output:0*
T0*+
_output_shapes
:@���������2
Tensordot_6/transpose�
Tensordot_6/ReshapeReshapeTensordot_6/transpose:y:0Tensordot_6/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot_6/Reshape�
Tensordot_6/MatMulMatMul"Tensordot_6/ReadVariableOp:value:0Tensordot_6/Reshape:output:0*
T0*(
_output_shapes
:����������2
Tensordot_6/MatMulu
Tensordot_6/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot_6/Const_2x
Tensordot_6/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_6/concat_1/axis�
Tensordot_6/concat_1ConcatV2Tensordot_6/Const_2:output:0Tensordot_6/GatherV2:output:0"Tensordot_6/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_6/concat_1�
Tensordot_6ReshapeTensordot_6/MatMul:product:0Tensordot_6/concat_1:output:0*
T0*,
_output_shapes
:����������2
Tensordot_6y
transpose_6/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_6/perm�
transpose_6	TransposeTensordot_6:output:0transpose_6/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_6j
add_4AddV2	sub_6:z:0transpose_6:y:0*
T0*,
_output_shapes
:����������2
add_4�
Tensordot_7/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot_7/ReadVariableOpn
Tensordot_7/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_7/axesu
Tensordot_7/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_7/freec
Tensordot_7/ShapeShapeImag:output:0*
T0*
_output_shapes
:2
Tensordot_7/Shapex
Tensordot_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_7/GatherV2/axis�
Tensordot_7/GatherV2GatherV2Tensordot_7/Shape:output:0Tensordot_7/free:output:0"Tensordot_7/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_7/GatherV2|
Tensordot_7/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_7/GatherV2_1/axis�
Tensordot_7/GatherV2_1GatherV2Tensordot_7/Shape:output:0Tensordot_7/axes:output:0$Tensordot_7/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_7/GatherV2_1p
Tensordot_7/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_7/Const�
Tensordot_7/ProdProdTensordot_7/GatherV2:output:0Tensordot_7/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_7/Prodt
Tensordot_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_7/Const_1�
Tensordot_7/Prod_1ProdTensordot_7/GatherV2_1:output:0Tensordot_7/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_7/Prod_1t
Tensordot_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_7/concat/axis�
Tensordot_7/concatConcatV2Tensordot_7/axes:output:0Tensordot_7/free:output:0 Tensordot_7/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_7/concat�
Tensordot_7/stackPackTensordot_7/Prod_1:output:0Tensordot_7/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_7/stack�
Tensordot_7/transpose	TransposeImag:output:0Tensordot_7/concat:output:0*
T0*+
_output_shapes
:@���������2
Tensordot_7/transpose�
Tensordot_7/ReshapeReshapeTensordot_7/transpose:y:0Tensordot_7/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot_7/Reshape�
Tensordot_7/MatMulMatMul"Tensordot_7/ReadVariableOp:value:0Tensordot_7/Reshape:output:0*
T0*(
_output_shapes
:����������2
Tensordot_7/MatMulu
Tensordot_7/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot_7/Const_2x
Tensordot_7/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_7/concat_1/axis�
Tensordot_7/concat_1ConcatV2Tensordot_7/Const_2:output:0Tensordot_7/GatherV2:output:0"Tensordot_7/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_7/concat_1�
Tensordot_7ReshapeTensordot_7/MatMul:product:0Tensordot_7/concat_1:output:0*
T0*,
_output_shapes
:����������2
Tensordot_7y
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm�
transpose_7	TransposeTensordot_7:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_7h
sub_7Sub	add_4:z:0transpose_7:y:0*
T0*,
_output_shapes
:����������2
sub_7l

norm_4/mulMul	sub_7:z:0	sub_7:z:0*
T0*,
_output_shapes
:����������2

norm_4/mul�
norm_4/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm_4/Sum/reduction_indices�

norm_4/SumSumnorm_4/mul:z:0%norm_4/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2

norm_4/Sumn
norm_4/SqrtSqrtnorm_4/Sum:output:0*
T0*,
_output_shapes
:����������2
norm_4/Sqrt�
norm_4/SqueezeSqueezenorm_4/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2
norm_4/Squeezef
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_4/dim�
ExpandDims_4
ExpandDimsnorm_4/Squeeze:output:0ExpandDims_4/dim:output:0*
T0*,
_output_shapes
:����������2
ExpandDims_4y
Tile_4/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile_4/multiples�
Tile_4TileExpandDims_4:output:0Tile_4/multiples:output:0*
T0*,
_output_shapes
:����������2
Tile_4�
Maximum_4/ReadVariableOpReadVariableOp!maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum_4/ReadVariableOp�
	Maximum_4MaximumTile_4:output:0 Maximum_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
	Maximum_4~
ReadVariableOp_4ReadVariableOp!maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_4�
	truediv_4RealDivReadVariableOp_4:value:0Maximum_4:z:0*
T0*,
_output_shapes
:����������2
	truediv_4W
sub_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
sub_8/xm
sub_8Subsub_8/x:output:0truediv_4:z:0*
T0*,
_output_shapes
:����������2
sub_8b
mul_4Mul	sub_8:z:0	sub_7:z:0*
T0*,
_output_shapes
:����������2
mul_4�
"conv1d_4/StatefulPartitionedCall_3StatefulPartitionedCall	mul_3:z:0conv1d_4_680018*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_6800092$
"conv1d_4/StatefulPartitionedCall_3�
"conv1d_5/StatefulPartitionedCall_3StatefulPartitionedCall	mul_4:z:0conv1d_5_680042*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_6800332$
"conv1d_5/StatefulPartitionedCall_3�
add_5AddV2+conv1d_4/StatefulPartitionedCall_3:output:0+conv1d_5/StatefulPartitionedCall_3:output:0*
T0*,
_output_shapes
:����������2
add_5�
Tensordot_8/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot_8/ReadVariableOpn
Tensordot_8/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_8/axesu
Tensordot_8/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_8/freec
Tensordot_8/ShapeShapeImag:output:0*
T0*
_output_shapes
:2
Tensordot_8/Shapex
Tensordot_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_8/GatherV2/axis�
Tensordot_8/GatherV2GatherV2Tensordot_8/Shape:output:0Tensordot_8/free:output:0"Tensordot_8/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_8/GatherV2|
Tensordot_8/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_8/GatherV2_1/axis�
Tensordot_8/GatherV2_1GatherV2Tensordot_8/Shape:output:0Tensordot_8/axes:output:0$Tensordot_8/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_8/GatherV2_1p
Tensordot_8/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_8/Const�
Tensordot_8/ProdProdTensordot_8/GatherV2:output:0Tensordot_8/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_8/Prodt
Tensordot_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_8/Const_1�
Tensordot_8/Prod_1ProdTensordot_8/GatherV2_1:output:0Tensordot_8/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_8/Prod_1t
Tensordot_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_8/concat/axis�
Tensordot_8/concatConcatV2Tensordot_8/axes:output:0Tensordot_8/free:output:0 Tensordot_8/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_8/concat�
Tensordot_8/stackPackTensordot_8/Prod_1:output:0Tensordot_8/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_8/stack�
Tensordot_8/transpose	TransposeImag:output:0Tensordot_8/concat:output:0*
T0*+
_output_shapes
:@���������2
Tensordot_8/transpose�
Tensordot_8/ReshapeReshapeTensordot_8/transpose:y:0Tensordot_8/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot_8/Reshape�
Tensordot_8/MatMulMatMul"Tensordot_8/ReadVariableOp:value:0Tensordot_8/Reshape:output:0*
T0*(
_output_shapes
:����������2
Tensordot_8/MatMulu
Tensordot_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot_8/Const_2x
Tensordot_8/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_8/concat_1/axis�
Tensordot_8/concat_1ConcatV2Tensordot_8/Const_2:output:0Tensordot_8/GatherV2:output:0"Tensordot_8/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_8/concat_1�
Tensordot_8ReshapeTensordot_8/MatMul:product:0Tensordot_8/concat_1:output:0*
T0*,
_output_shapes
:����������2
Tensordot_8y
transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_8/perm�
transpose_8	TransposeTensordot_8:output:0transpose_8/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_8j
add_6AddV2	add_5:z:0transpose_8:y:0*
T0*,
_output_shapes
:����������2
add_6�
Tensordot_9/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot_9/ReadVariableOpn
Tensordot_9/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_9/axesu
Tensordot_9/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_9/freec
Tensordot_9/ShapeShapeReal:output:0*
T0*
_output_shapes
:2
Tensordot_9/Shapex
Tensordot_9/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_9/GatherV2/axis�
Tensordot_9/GatherV2GatherV2Tensordot_9/Shape:output:0Tensordot_9/free:output:0"Tensordot_9/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_9/GatherV2|
Tensordot_9/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_9/GatherV2_1/axis�
Tensordot_9/GatherV2_1GatherV2Tensordot_9/Shape:output:0Tensordot_9/axes:output:0$Tensordot_9/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_9/GatherV2_1p
Tensordot_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_9/Const�
Tensordot_9/ProdProdTensordot_9/GatherV2:output:0Tensordot_9/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_9/Prodt
Tensordot_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_9/Const_1�
Tensordot_9/Prod_1ProdTensordot_9/GatherV2_1:output:0Tensordot_9/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_9/Prod_1t
Tensordot_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_9/concat/axis�
Tensordot_9/concatConcatV2Tensordot_9/axes:output:0Tensordot_9/free:output:0 Tensordot_9/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_9/concat�
Tensordot_9/stackPackTensordot_9/Prod_1:output:0Tensordot_9/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_9/stack�
Tensordot_9/transpose	TransposeReal:output:0Tensordot_9/concat:output:0*
T0*+
_output_shapes
:@���������2
Tensordot_9/transpose�
Tensordot_9/ReshapeReshapeTensordot_9/transpose:y:0Tensordot_9/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot_9/Reshape�
Tensordot_9/MatMulMatMul"Tensordot_9/ReadVariableOp:value:0Tensordot_9/Reshape:output:0*
T0*(
_output_shapes
:����������2
Tensordot_9/MatMulu
Tensordot_9/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot_9/Const_2x
Tensordot_9/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_9/concat_1/axis�
Tensordot_9/concat_1ConcatV2Tensordot_9/Const_2:output:0Tensordot_9/GatherV2:output:0"Tensordot_9/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_9/concat_1�
Tensordot_9ReshapeTensordot_9/MatMul:product:0Tensordot_9/concat_1:output:0*
T0*,
_output_shapes
:����������2
Tensordot_9y
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm�
transpose_9	TransposeTensordot_9:output:0transpose_9/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_9j
add_7AddV2	add_6:z:0transpose_9:y:0*
T0*,
_output_shapes
:����������2
add_7l

norm_5/mulMul	add_7:z:0	add_7:z:0*
T0*,
_output_shapes
:����������2

norm_5/mul�
norm_5/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm_5/Sum/reduction_indices�

norm_5/SumSumnorm_5/mul:z:0%norm_5/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2

norm_5/Sumn
norm_5/SqrtSqrtnorm_5/Sum:output:0*
T0*,
_output_shapes
:����������2
norm_5/Sqrt�
norm_5/SqueezeSqueezenorm_5/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2
norm_5/Squeezef
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_5/dim�
ExpandDims_5
ExpandDimsnorm_5/Squeeze:output:0ExpandDims_5/dim:output:0*
T0*,
_output_shapes
:����������2
ExpandDims_5y
Tile_5/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile_5/multiples�
Tile_5TileExpandDims_5:output:0Tile_5/multiples:output:0*
T0*,
_output_shapes
:����������2
Tile_5�
Maximum_5/ReadVariableOpReadVariableOp!maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum_5/ReadVariableOp�
	Maximum_5MaximumTile_5:output:0 Maximum_5/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
	Maximum_5~
ReadVariableOp_5ReadVariableOp!maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_5�
	truediv_5RealDivReadVariableOp_5:value:0Maximum_5:z:0*
T0*,
_output_shapes
:����������2
	truediv_5W
sub_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
sub_9/xm
sub_9Subsub_9/x:output:0truediv_5:z:0*
T0*,
_output_shapes
:����������2
sub_9b
mul_5Mul	sub_9:z:0	add_7:z:0*
T0*,
_output_shapes
:����������2
mul_5\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2	mul_4:z:0	mul_5:z:0concat/axis:output:0*
N*
T0*,
_output_shapes
:����������2
concat�
IdentityIdentityconcat:output:0!^conv1d_4/StatefulPartitionedCall#^conv1d_4/StatefulPartitionedCall_1#^conv1d_4/StatefulPartitionedCall_2#^conv1d_4/StatefulPartitionedCall_3!^conv1d_5/StatefulPartitionedCall#^conv1d_5/StatefulPartitionedCall_1#^conv1d_5/StatefulPartitionedCall_2#^conv1d_5/StatefulPartitionedCall_3*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������@:::::::2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2H
"conv1d_4/StatefulPartitionedCall_1"conv1d_4/StatefulPartitionedCall_12H
"conv1d_4/StatefulPartitionedCall_2"conv1d_4/StatefulPartitionedCall_22H
"conv1d_4/StatefulPartitionedCall_3"conv1d_4/StatefulPartitionedCall_32D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2H
"conv1d_5/StatefulPartitionedCall_1"conv1d_5/StatefulPartitionedCall_12H
"conv1d_5/StatefulPartitionedCall_2"conv1d_5/StatefulPartitionedCall_22H
"conv1d_5/StatefulPartitionedCall_3"conv1d_5/StatefulPartitionedCall_3:T P
+
_output_shapes
:���������@
!
_user_specified_name	input_1
�
�
$__inference_signature_wrapper_680379
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_6799082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������@:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������@
!
_user_specified_name	input_1
�
o
)__inference_conv1d_5_layer_call_fn_680417

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_6800332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv1d_5_layer_call_and_return_conditional_losses_680033

inputs/
+conv1d_expanddims_1_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�>
�
__inference__traced_save_680527
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop@
<savev2_lista__toeplitz_2_conv1d_4_kernel_read_readvariableop@
<savev2_lista__toeplitz_2_conv1d_5_kernel_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_4_read_readvariableop)
%savev2_variable_5_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop.
*savev2_adam_variable_m_read_readvariableop0
,savev2_adam_variable_m_1_read_readvariableopG
Csavev2_adam_lista__toeplitz_2_conv1d_4_kernel_m_read_readvariableopG
Csavev2_adam_lista__toeplitz_2_conv1d_5_kernel_m_read_readvariableop0
,savev2_adam_variable_m_2_read_readvariableop0
,savev2_adam_variable_m_3_read_readvariableop0
,savev2_adam_variable_m_4_read_readvariableop.
*savev2_adam_variable_v_read_readvariableop0
,savev2_adam_variable_v_1_read_readvariableopG
Csavev2_adam_lista__toeplitz_2_conv1d_4_kernel_v_read_readvariableopG
Csavev2_adam_lista__toeplitz_2_conv1d_5_kernel_v_read_readvariableop0
,savev2_adam_variable_v_2_read_readvariableop0
,savev2_adam_variable_v_3_read_readvariableop0
,savev2_adam_variable_v_4_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_35fb63e4666844699b242ab98c355a70/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�BWe_r/.ATTRIBUTES/VARIABLE_VALUEBWe_i/.ATTRIBUTES/VARIABLE_VALUEB alpha/.ATTRIBUTES/VARIABLE_VALUEB&hg_r/kernel/.ATTRIBUTES/VARIABLE_VALUEB&hg_i/kernel/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/0/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/1/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/2/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB;We_r/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;We_i/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBhg_r/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBhg_i/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;We_r/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB;We_i/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBhg_r/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBhg_i/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop<savev2_lista__toeplitz_2_conv1d_4_kernel_read_readvariableop<savev2_lista__toeplitz_2_conv1d_5_kernel_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_5_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop*savev2_adam_variable_m_read_readvariableop,savev2_adam_variable_m_1_read_readvariableopCsavev2_adam_lista__toeplitz_2_conv1d_4_kernel_m_read_readvariableopCsavev2_adam_lista__toeplitz_2_conv1d_5_kernel_m_read_readvariableop,savev2_adam_variable_m_2_read_readvariableop,savev2_adam_variable_m_3_read_readvariableop,savev2_adam_variable_m_4_read_readvariableop*savev2_adam_variable_v_read_readvariableop,savev2_adam_variable_v_1_read_readvariableopCsavev2_adam_lista__toeplitz_2_conv1d_4_kernel_v_read_readvariableopCsavev2_adam_lista__toeplitz_2_conv1d_5_kernel_v_read_readvariableop,savev2_adam_variable_v_2_read_readvariableop,savev2_adam_variable_v_3_read_readvariableop,savev2_adam_variable_v_4_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�@:	�@: :�:�: : : : : : : : : : :	�@:	�@:�:�: : : :	�@:	�@:�:�: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :)%
#
_output_shapes
:�:)%
#
_output_shapes
:�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:)%
#
_output_shapes
:�:)%
#
_output_shapes
:�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:)%
#
_output_shapes
:�:)%
#
_output_shapes
:�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
o
)__inference_conv1d_4_layer_call_fn_680398

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_6800092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
2__inference_lista__toeplitz_2_layer_call_fn_680350
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_lista__toeplitz_2_layer_call_and_return_conditional_losses_6803302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������@:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������@
!
_user_specified_name	input_1
�
�
D__inference_conv1d_4_layer_call_and_return_conditional_losses_680391

inputs/
+conv1d_expanddims_1_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv1d_5_layer_call_and_return_conditional_losses_680410

inputs/
+conv1d_expanddims_1_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_679908
input_17
3lista__toeplitz_2_tensordot_readvariableop_resource5
1lista__toeplitz_2_maximum_readvariableop_resource9
5lista__toeplitz_2_tensordot_1_readvariableop_resourceJ
Flista__toeplitz_2_conv1d_4_conv1d_expanddims_1_readvariableop_resourceJ
Flista__toeplitz_2_conv1d_5_conv1d_expanddims_1_readvariableop_resource7
3lista__toeplitz_2_maximum_2_readvariableop_resource7
3lista__toeplitz_2_maximum_4_readvariableop_resource
identity�n
lista__toeplitz_2/RealRealinput_1*+
_output_shapes
:���������@2
lista__toeplitz_2/Realn
lista__toeplitz_2/ImagImaginput_1*+
_output_shapes
:���������@2
lista__toeplitz_2/Imag�
*lista__toeplitz_2/Tensordot/ReadVariableOpReadVariableOp3lista__toeplitz_2_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02,
*lista__toeplitz_2/Tensordot/ReadVariableOp�
 lista__toeplitz_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 lista__toeplitz_2/Tensordot/axes�
 lista__toeplitz_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2"
 lista__toeplitz_2/Tensordot/free�
!lista__toeplitz_2/Tensordot/ShapeShapelista__toeplitz_2/Real:output:0*
T0*
_output_shapes
:2#
!lista__toeplitz_2/Tensordot/Shape�
)lista__toeplitz_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot/GatherV2/axis�
$lista__toeplitz_2/Tensordot/GatherV2GatherV2*lista__toeplitz_2/Tensordot/Shape:output:0)lista__toeplitz_2/Tensordot/free:output:02lista__toeplitz_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot/GatherV2�
+lista__toeplitz_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot/GatherV2_1/axis�
&lista__toeplitz_2/Tensordot/GatherV2_1GatherV2*lista__toeplitz_2/Tensordot/Shape:output:0)lista__toeplitz_2/Tensordot/axes:output:04lista__toeplitz_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot/GatherV2_1�
!lista__toeplitz_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!lista__toeplitz_2/Tensordot/Const�
 lista__toeplitz_2/Tensordot/ProdProd-lista__toeplitz_2/Tensordot/GatherV2:output:0*lista__toeplitz_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 lista__toeplitz_2/Tensordot/Prod�
#lista__toeplitz_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#lista__toeplitz_2/Tensordot/Const_1�
"lista__toeplitz_2/Tensordot/Prod_1Prod/lista__toeplitz_2/Tensordot/GatherV2_1:output:0,lista__toeplitz_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"lista__toeplitz_2/Tensordot/Prod_1�
'lista__toeplitz_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'lista__toeplitz_2/Tensordot/concat/axis�
"lista__toeplitz_2/Tensordot/concatConcatV2)lista__toeplitz_2/Tensordot/axes:output:0)lista__toeplitz_2/Tensordot/free:output:00lista__toeplitz_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"lista__toeplitz_2/Tensordot/concat�
!lista__toeplitz_2/Tensordot/stackPack+lista__toeplitz_2/Tensordot/Prod_1:output:0)lista__toeplitz_2/Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:2#
!lista__toeplitz_2/Tensordot/stack�
%lista__toeplitz_2/Tensordot/transpose	Transposelista__toeplitz_2/Real:output:0+lista__toeplitz_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:@���������2'
%lista__toeplitz_2/Tensordot/transpose�
#lista__toeplitz_2/Tensordot/ReshapeReshape)lista__toeplitz_2/Tensordot/transpose:y:0*lista__toeplitz_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2%
#lista__toeplitz_2/Tensordot/Reshape�
"lista__toeplitz_2/Tensordot/MatMulMatMul2lista__toeplitz_2/Tensordot/ReadVariableOp:value:0,lista__toeplitz_2/Tensordot/Reshape:output:0*
T0*(
_output_shapes
:����������2$
"lista__toeplitz_2/Tensordot/MatMul�
#lista__toeplitz_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2%
#lista__toeplitz_2/Tensordot/Const_2�
)lista__toeplitz_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot/concat_1/axis�
$lista__toeplitz_2/Tensordot/concat_1ConcatV2,lista__toeplitz_2/Tensordot/Const_2:output:0-lista__toeplitz_2/Tensordot/GatherV2:output:02lista__toeplitz_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot/concat_1�
lista__toeplitz_2/TensordotReshape,lista__toeplitz_2/Tensordot/MatMul:product:0-lista__toeplitz_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tensordot�
 lista__toeplitz_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 lista__toeplitz_2/transpose/perm�
lista__toeplitz_2/transpose	Transpose$lista__toeplitz_2/Tensordot:output:0)lista__toeplitz_2/transpose/perm:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/transpose�
lista__toeplitz_2/norm/mulMullista__toeplitz_2/transpose:y:0lista__toeplitz_2/transpose:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm/mul�
,lista__toeplitz_2/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2.
,lista__toeplitz_2/norm/Sum/reduction_indices�
lista__toeplitz_2/norm/SumSumlista__toeplitz_2/norm/mul:z:05lista__toeplitz_2/norm/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2
lista__toeplitz_2/norm/Sum�
lista__toeplitz_2/norm/SqrtSqrt#lista__toeplitz_2/norm/Sum:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm/Sqrt�
lista__toeplitz_2/norm/SqueezeSqueezelista__toeplitz_2/norm/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2 
lista__toeplitz_2/norm/Squeeze�
 lista__toeplitz_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 lista__toeplitz_2/ExpandDims/dim�
lista__toeplitz_2/ExpandDims
ExpandDims'lista__toeplitz_2/norm/Squeeze:output:0)lista__toeplitz_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/ExpandDims�
 lista__toeplitz_2/Tile/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2"
 lista__toeplitz_2/Tile/multiples�
lista__toeplitz_2/TileTile%lista__toeplitz_2/ExpandDims:output:0)lista__toeplitz_2/Tile/multiples:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tile�
(lista__toeplitz_2/Maximum/ReadVariableOpReadVariableOp1lista__toeplitz_2_maximum_readvariableop_resource*
_output_shapes
: *
dtype02*
(lista__toeplitz_2/Maximum/ReadVariableOp�
lista__toeplitz_2/MaximumMaximumlista__toeplitz_2/Tile:output:00lista__toeplitz_2/Maximum/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Maximum�
 lista__toeplitz_2/ReadVariableOpReadVariableOp1lista__toeplitz_2_maximum_readvariableop_resource*
_output_shapes
: *
dtype02"
 lista__toeplitz_2/ReadVariableOp�
lista__toeplitz_2/truedivRealDiv(lista__toeplitz_2/ReadVariableOp:value:0lista__toeplitz_2/Maximum:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/truedivw
lista__toeplitz_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
lista__toeplitz_2/sub/x�
lista__toeplitz_2/subSub lista__toeplitz_2/sub/x:output:0lista__toeplitz_2/truediv:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/sub�
lista__toeplitz_2/mulMullista__toeplitz_2/sub:z:0lista__toeplitz_2/transpose:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/mul�
,lista__toeplitz_2/Tensordot_1/ReadVariableOpReadVariableOp5lista__toeplitz_2_tensordot_1_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,lista__toeplitz_2/Tensordot_1/ReadVariableOp�
"lista__toeplitz_2/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"lista__toeplitz_2/Tensordot_1/axes�
"lista__toeplitz_2/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"lista__toeplitz_2/Tensordot_1/free�
#lista__toeplitz_2/Tensordot_1/ShapeShapelista__toeplitz_2/Imag:output:0*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_1/Shape�
+lista__toeplitz_2/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_1/GatherV2/axis�
&lista__toeplitz_2/Tensordot_1/GatherV2GatherV2,lista__toeplitz_2/Tensordot_1/Shape:output:0+lista__toeplitz_2/Tensordot_1/free:output:04lista__toeplitz_2/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_1/GatherV2�
-lista__toeplitz_2/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-lista__toeplitz_2/Tensordot_1/GatherV2_1/axis�
(lista__toeplitz_2/Tensordot_1/GatherV2_1GatherV2,lista__toeplitz_2/Tensordot_1/Shape:output:0+lista__toeplitz_2/Tensordot_1/axes:output:06lista__toeplitz_2/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(lista__toeplitz_2/Tensordot_1/GatherV2_1�
#lista__toeplitz_2/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#lista__toeplitz_2/Tensordot_1/Const�
"lista__toeplitz_2/Tensordot_1/ProdProd/lista__toeplitz_2/Tensordot_1/GatherV2:output:0,lista__toeplitz_2/Tensordot_1/Const:output:0*
T0*
_output_shapes
: 2$
"lista__toeplitz_2/Tensordot_1/Prod�
%lista__toeplitz_2/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%lista__toeplitz_2/Tensordot_1/Const_1�
$lista__toeplitz_2/Tensordot_1/Prod_1Prod1lista__toeplitz_2/Tensordot_1/GatherV2_1:output:0.lista__toeplitz_2/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: 2&
$lista__toeplitz_2/Tensordot_1/Prod_1�
)lista__toeplitz_2/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot_1/concat/axis�
$lista__toeplitz_2/Tensordot_1/concatConcatV2+lista__toeplitz_2/Tensordot_1/axes:output:0+lista__toeplitz_2/Tensordot_1/free:output:02lista__toeplitz_2/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot_1/concat�
#lista__toeplitz_2/Tensordot_1/stackPack-lista__toeplitz_2/Tensordot_1/Prod_1:output:0+lista__toeplitz_2/Tensordot_1/Prod:output:0*
N*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_1/stack�
'lista__toeplitz_2/Tensordot_1/transpose	Transposelista__toeplitz_2/Imag:output:0-lista__toeplitz_2/Tensordot_1/concat:output:0*
T0*+
_output_shapes
:@���������2)
'lista__toeplitz_2/Tensordot_1/transpose�
%lista__toeplitz_2/Tensordot_1/ReshapeReshape+lista__toeplitz_2/Tensordot_1/transpose:y:0,lista__toeplitz_2/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:������������������2'
%lista__toeplitz_2/Tensordot_1/Reshape�
$lista__toeplitz_2/Tensordot_1/MatMulMatMul4lista__toeplitz_2/Tensordot_1/ReadVariableOp:value:0.lista__toeplitz_2/Tensordot_1/Reshape:output:0*
T0*(
_output_shapes
:����������2&
$lista__toeplitz_2/Tensordot_1/MatMul�
%lista__toeplitz_2/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%lista__toeplitz_2/Tensordot_1/Const_2�
+lista__toeplitz_2/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_1/concat_1/axis�
&lista__toeplitz_2/Tensordot_1/concat_1ConcatV2.lista__toeplitz_2/Tensordot_1/Const_2:output:0/lista__toeplitz_2/Tensordot_1/GatherV2:output:04lista__toeplitz_2/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_1/concat_1�
lista__toeplitz_2/Tensordot_1Reshape.lista__toeplitz_2/Tensordot_1/MatMul:product:0/lista__toeplitz_2/Tensordot_1/concat_1:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tensordot_1�
"lista__toeplitz_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"lista__toeplitz_2/transpose_1/perm�
lista__toeplitz_2/transpose_1	Transpose&lista__toeplitz_2/Tensordot_1:output:0+lista__toeplitz_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/transpose_1�
lista__toeplitz_2/norm_1/mulMul!lista__toeplitz_2/transpose_1:y:0!lista__toeplitz_2/transpose_1:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm_1/mul�
.lista__toeplitz_2/norm_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:20
.lista__toeplitz_2/norm_1/Sum/reduction_indices�
lista__toeplitz_2/norm_1/SumSum lista__toeplitz_2/norm_1/mul:z:07lista__toeplitz_2/norm_1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2
lista__toeplitz_2/norm_1/Sum�
lista__toeplitz_2/norm_1/SqrtSqrt%lista__toeplitz_2/norm_1/Sum:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm_1/Sqrt�
 lista__toeplitz_2/norm_1/SqueezeSqueeze!lista__toeplitz_2/norm_1/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2"
 lista__toeplitz_2/norm_1/Squeeze�
"lista__toeplitz_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lista__toeplitz_2/ExpandDims_1/dim�
lista__toeplitz_2/ExpandDims_1
ExpandDims)lista__toeplitz_2/norm_1/Squeeze:output:0+lista__toeplitz_2/ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:����������2 
lista__toeplitz_2/ExpandDims_1�
"lista__toeplitz_2/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2$
"lista__toeplitz_2/Tile_1/multiples�
lista__toeplitz_2/Tile_1Tile'lista__toeplitz_2/ExpandDims_1:output:0+lista__toeplitz_2/Tile_1/multiples:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tile_1�
*lista__toeplitz_2/Maximum_1/ReadVariableOpReadVariableOp1lista__toeplitz_2_maximum_readvariableop_resource*
_output_shapes
: *
dtype02,
*lista__toeplitz_2/Maximum_1/ReadVariableOp�
lista__toeplitz_2/Maximum_1Maximum!lista__toeplitz_2/Tile_1:output:02lista__toeplitz_2/Maximum_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Maximum_1�
"lista__toeplitz_2/ReadVariableOp_1ReadVariableOp1lista__toeplitz_2_maximum_readvariableop_resource*
_output_shapes
: *
dtype02$
"lista__toeplitz_2/ReadVariableOp_1�
lista__toeplitz_2/truediv_1RealDiv*lista__toeplitz_2/ReadVariableOp_1:value:0lista__toeplitz_2/Maximum_1:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/truediv_1{
lista__toeplitz_2/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
lista__toeplitz_2/sub_1/x�
lista__toeplitz_2/sub_1Sub"lista__toeplitz_2/sub_1/x:output:0lista__toeplitz_2/truediv_1:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/sub_1�
lista__toeplitz_2/mul_1Mullista__toeplitz_2/sub_1:z:0!lista__toeplitz_2/transpose_1:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/mul_1�
0lista__toeplitz_2/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0lista__toeplitz_2/conv1d_4/conv1d/ExpandDims/dim�
,lista__toeplitz_2/conv1d_4/conv1d/ExpandDims
ExpandDimslista__toeplitz_2/mul:z:09lista__toeplitz_2/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2.
,lista__toeplitz_2/conv1d_4/conv1d/ExpandDims�
=lista__toeplitz_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFlista__toeplitz_2_conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02?
=lista__toeplitz_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�
2lista__toeplitz_2/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2lista__toeplitz_2/conv1d_4/conv1d/ExpandDims_1/dim�
.lista__toeplitz_2/conv1d_4/conv1d/ExpandDims_1
ExpandDimsElista__toeplitz_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0;lista__toeplitz_2/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�20
.lista__toeplitz_2/conv1d_4/conv1d/ExpandDims_1�
!lista__toeplitz_2/conv1d_4/conv1dConv2D5lista__toeplitz_2/conv1d_4/conv1d/ExpandDims:output:07lista__toeplitz_2/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2#
!lista__toeplitz_2/conv1d_4/conv1d�
)lista__toeplitz_2/conv1d_4/conv1d/SqueezeSqueeze*lista__toeplitz_2/conv1d_4/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2+
)lista__toeplitz_2/conv1d_4/conv1d/Squeeze�
0lista__toeplitz_2/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0lista__toeplitz_2/conv1d_5/conv1d/ExpandDims/dim�
,lista__toeplitz_2/conv1d_5/conv1d/ExpandDims
ExpandDimslista__toeplitz_2/mul_1:z:09lista__toeplitz_2/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2.
,lista__toeplitz_2/conv1d_5/conv1d/ExpandDims�
=lista__toeplitz_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFlista__toeplitz_2_conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02?
=lista__toeplitz_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�
2lista__toeplitz_2/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2lista__toeplitz_2/conv1d_5/conv1d/ExpandDims_1/dim�
.lista__toeplitz_2/conv1d_5/conv1d/ExpandDims_1
ExpandDimsElista__toeplitz_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0;lista__toeplitz_2/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�20
.lista__toeplitz_2/conv1d_5/conv1d/ExpandDims_1�
!lista__toeplitz_2/conv1d_5/conv1dConv2D5lista__toeplitz_2/conv1d_5/conv1d/ExpandDims:output:07lista__toeplitz_2/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2#
!lista__toeplitz_2/conv1d_5/conv1d�
)lista__toeplitz_2/conv1d_5/conv1d/SqueezeSqueeze*lista__toeplitz_2/conv1d_5/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2+
)lista__toeplitz_2/conv1d_5/conv1d/Squeeze�
lista__toeplitz_2/sub_2Sub2lista__toeplitz_2/conv1d_4/conv1d/Squeeze:output:02lista__toeplitz_2/conv1d_5/conv1d/Squeeze:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/sub_2�
,lista__toeplitz_2/Tensordot_2/ReadVariableOpReadVariableOp3lista__toeplitz_2_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,lista__toeplitz_2/Tensordot_2/ReadVariableOp�
"lista__toeplitz_2/Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"lista__toeplitz_2/Tensordot_2/axes�
"lista__toeplitz_2/Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"lista__toeplitz_2/Tensordot_2/free�
#lista__toeplitz_2/Tensordot_2/ShapeShapelista__toeplitz_2/Real:output:0*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_2/Shape�
+lista__toeplitz_2/Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_2/GatherV2/axis�
&lista__toeplitz_2/Tensordot_2/GatherV2GatherV2,lista__toeplitz_2/Tensordot_2/Shape:output:0+lista__toeplitz_2/Tensordot_2/free:output:04lista__toeplitz_2/Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_2/GatherV2�
-lista__toeplitz_2/Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-lista__toeplitz_2/Tensordot_2/GatherV2_1/axis�
(lista__toeplitz_2/Tensordot_2/GatherV2_1GatherV2,lista__toeplitz_2/Tensordot_2/Shape:output:0+lista__toeplitz_2/Tensordot_2/axes:output:06lista__toeplitz_2/Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(lista__toeplitz_2/Tensordot_2/GatherV2_1�
#lista__toeplitz_2/Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#lista__toeplitz_2/Tensordot_2/Const�
"lista__toeplitz_2/Tensordot_2/ProdProd/lista__toeplitz_2/Tensordot_2/GatherV2:output:0,lista__toeplitz_2/Tensordot_2/Const:output:0*
T0*
_output_shapes
: 2$
"lista__toeplitz_2/Tensordot_2/Prod�
%lista__toeplitz_2/Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%lista__toeplitz_2/Tensordot_2/Const_1�
$lista__toeplitz_2/Tensordot_2/Prod_1Prod1lista__toeplitz_2/Tensordot_2/GatherV2_1:output:0.lista__toeplitz_2/Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: 2&
$lista__toeplitz_2/Tensordot_2/Prod_1�
)lista__toeplitz_2/Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot_2/concat/axis�
$lista__toeplitz_2/Tensordot_2/concatConcatV2+lista__toeplitz_2/Tensordot_2/axes:output:0+lista__toeplitz_2/Tensordot_2/free:output:02lista__toeplitz_2/Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot_2/concat�
#lista__toeplitz_2/Tensordot_2/stackPack-lista__toeplitz_2/Tensordot_2/Prod_1:output:0+lista__toeplitz_2/Tensordot_2/Prod:output:0*
N*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_2/stack�
'lista__toeplitz_2/Tensordot_2/transpose	Transposelista__toeplitz_2/Real:output:0-lista__toeplitz_2/Tensordot_2/concat:output:0*
T0*+
_output_shapes
:@���������2)
'lista__toeplitz_2/Tensordot_2/transpose�
%lista__toeplitz_2/Tensordot_2/ReshapeReshape+lista__toeplitz_2/Tensordot_2/transpose:y:0,lista__toeplitz_2/Tensordot_2/stack:output:0*
T0*0
_output_shapes
:������������������2'
%lista__toeplitz_2/Tensordot_2/Reshape�
$lista__toeplitz_2/Tensordot_2/MatMulMatMul4lista__toeplitz_2/Tensordot_2/ReadVariableOp:value:0.lista__toeplitz_2/Tensordot_2/Reshape:output:0*
T0*(
_output_shapes
:����������2&
$lista__toeplitz_2/Tensordot_2/MatMul�
%lista__toeplitz_2/Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%lista__toeplitz_2/Tensordot_2/Const_2�
+lista__toeplitz_2/Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_2/concat_1/axis�
&lista__toeplitz_2/Tensordot_2/concat_1ConcatV2.lista__toeplitz_2/Tensordot_2/Const_2:output:0/lista__toeplitz_2/Tensordot_2/GatherV2:output:04lista__toeplitz_2/Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_2/concat_1�
lista__toeplitz_2/Tensordot_2Reshape.lista__toeplitz_2/Tensordot_2/MatMul:product:0/lista__toeplitz_2/Tensordot_2/concat_1:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tensordot_2�
"lista__toeplitz_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"lista__toeplitz_2/transpose_2/perm�
lista__toeplitz_2/transpose_2	Transpose&lista__toeplitz_2/Tensordot_2:output:0+lista__toeplitz_2/transpose_2/perm:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/transpose_2�
lista__toeplitz_2/addAddV2lista__toeplitz_2/sub_2:z:0!lista__toeplitz_2/transpose_2:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/add�
,lista__toeplitz_2/Tensordot_3/ReadVariableOpReadVariableOp5lista__toeplitz_2_tensordot_1_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,lista__toeplitz_2/Tensordot_3/ReadVariableOp�
"lista__toeplitz_2/Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"lista__toeplitz_2/Tensordot_3/axes�
"lista__toeplitz_2/Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"lista__toeplitz_2/Tensordot_3/free�
#lista__toeplitz_2/Tensordot_3/ShapeShapelista__toeplitz_2/Imag:output:0*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_3/Shape�
+lista__toeplitz_2/Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_3/GatherV2/axis�
&lista__toeplitz_2/Tensordot_3/GatherV2GatherV2,lista__toeplitz_2/Tensordot_3/Shape:output:0+lista__toeplitz_2/Tensordot_3/free:output:04lista__toeplitz_2/Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_3/GatherV2�
-lista__toeplitz_2/Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-lista__toeplitz_2/Tensordot_3/GatherV2_1/axis�
(lista__toeplitz_2/Tensordot_3/GatherV2_1GatherV2,lista__toeplitz_2/Tensordot_3/Shape:output:0+lista__toeplitz_2/Tensordot_3/axes:output:06lista__toeplitz_2/Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(lista__toeplitz_2/Tensordot_3/GatherV2_1�
#lista__toeplitz_2/Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#lista__toeplitz_2/Tensordot_3/Const�
"lista__toeplitz_2/Tensordot_3/ProdProd/lista__toeplitz_2/Tensordot_3/GatherV2:output:0,lista__toeplitz_2/Tensordot_3/Const:output:0*
T0*
_output_shapes
: 2$
"lista__toeplitz_2/Tensordot_3/Prod�
%lista__toeplitz_2/Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%lista__toeplitz_2/Tensordot_3/Const_1�
$lista__toeplitz_2/Tensordot_3/Prod_1Prod1lista__toeplitz_2/Tensordot_3/GatherV2_1:output:0.lista__toeplitz_2/Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: 2&
$lista__toeplitz_2/Tensordot_3/Prod_1�
)lista__toeplitz_2/Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot_3/concat/axis�
$lista__toeplitz_2/Tensordot_3/concatConcatV2+lista__toeplitz_2/Tensordot_3/axes:output:0+lista__toeplitz_2/Tensordot_3/free:output:02lista__toeplitz_2/Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot_3/concat�
#lista__toeplitz_2/Tensordot_3/stackPack-lista__toeplitz_2/Tensordot_3/Prod_1:output:0+lista__toeplitz_2/Tensordot_3/Prod:output:0*
N*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_3/stack�
'lista__toeplitz_2/Tensordot_3/transpose	Transposelista__toeplitz_2/Imag:output:0-lista__toeplitz_2/Tensordot_3/concat:output:0*
T0*+
_output_shapes
:@���������2)
'lista__toeplitz_2/Tensordot_3/transpose�
%lista__toeplitz_2/Tensordot_3/ReshapeReshape+lista__toeplitz_2/Tensordot_3/transpose:y:0,lista__toeplitz_2/Tensordot_3/stack:output:0*
T0*0
_output_shapes
:������������������2'
%lista__toeplitz_2/Tensordot_3/Reshape�
$lista__toeplitz_2/Tensordot_3/MatMulMatMul4lista__toeplitz_2/Tensordot_3/ReadVariableOp:value:0.lista__toeplitz_2/Tensordot_3/Reshape:output:0*
T0*(
_output_shapes
:����������2&
$lista__toeplitz_2/Tensordot_3/MatMul�
%lista__toeplitz_2/Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%lista__toeplitz_2/Tensordot_3/Const_2�
+lista__toeplitz_2/Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_3/concat_1/axis�
&lista__toeplitz_2/Tensordot_3/concat_1ConcatV2.lista__toeplitz_2/Tensordot_3/Const_2:output:0/lista__toeplitz_2/Tensordot_3/GatherV2:output:04lista__toeplitz_2/Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_3/concat_1�
lista__toeplitz_2/Tensordot_3Reshape.lista__toeplitz_2/Tensordot_3/MatMul:product:0/lista__toeplitz_2/Tensordot_3/concat_1:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tensordot_3�
"lista__toeplitz_2/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"lista__toeplitz_2/transpose_3/perm�
lista__toeplitz_2/transpose_3	Transpose&lista__toeplitz_2/Tensordot_3:output:0+lista__toeplitz_2/transpose_3/perm:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/transpose_3�
lista__toeplitz_2/sub_3Sublista__toeplitz_2/add:z:0!lista__toeplitz_2/transpose_3:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/sub_3�
lista__toeplitz_2/norm_2/mulMullista__toeplitz_2/sub_3:z:0lista__toeplitz_2/sub_3:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm_2/mul�
.lista__toeplitz_2/norm_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:20
.lista__toeplitz_2/norm_2/Sum/reduction_indices�
lista__toeplitz_2/norm_2/SumSum lista__toeplitz_2/norm_2/mul:z:07lista__toeplitz_2/norm_2/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2
lista__toeplitz_2/norm_2/Sum�
lista__toeplitz_2/norm_2/SqrtSqrt%lista__toeplitz_2/norm_2/Sum:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm_2/Sqrt�
 lista__toeplitz_2/norm_2/SqueezeSqueeze!lista__toeplitz_2/norm_2/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2"
 lista__toeplitz_2/norm_2/Squeeze�
"lista__toeplitz_2/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lista__toeplitz_2/ExpandDims_2/dim�
lista__toeplitz_2/ExpandDims_2
ExpandDims)lista__toeplitz_2/norm_2/Squeeze:output:0+lista__toeplitz_2/ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:����������2 
lista__toeplitz_2/ExpandDims_2�
"lista__toeplitz_2/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2$
"lista__toeplitz_2/Tile_2/multiples�
lista__toeplitz_2/Tile_2Tile'lista__toeplitz_2/ExpandDims_2:output:0+lista__toeplitz_2/Tile_2/multiples:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tile_2�
*lista__toeplitz_2/Maximum_2/ReadVariableOpReadVariableOp3lista__toeplitz_2_maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02,
*lista__toeplitz_2/Maximum_2/ReadVariableOp�
lista__toeplitz_2/Maximum_2Maximum!lista__toeplitz_2/Tile_2:output:02lista__toeplitz_2/Maximum_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Maximum_2�
"lista__toeplitz_2/ReadVariableOp_2ReadVariableOp3lista__toeplitz_2_maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02$
"lista__toeplitz_2/ReadVariableOp_2�
lista__toeplitz_2/truediv_2RealDiv*lista__toeplitz_2/ReadVariableOp_2:value:0lista__toeplitz_2/Maximum_2:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/truediv_2{
lista__toeplitz_2/sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
lista__toeplitz_2/sub_4/x�
lista__toeplitz_2/sub_4Sub"lista__toeplitz_2/sub_4/x:output:0lista__toeplitz_2/truediv_2:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/sub_4�
lista__toeplitz_2/mul_2Mullista__toeplitz_2/sub_4:z:0lista__toeplitz_2/sub_3:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/mul_2�
2lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims/dim�
.lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims
ExpandDimslista__toeplitz_2/mul_1:z:0;lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������20
.lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims�
?lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpFlista__toeplitz_2_conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02A
?lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims_1/ReadVariableOp�
4lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims_1/dim�
0lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims_1
ExpandDimsGlista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims_1/ReadVariableOp:value:0=lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�22
0lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims_1�
#lista__toeplitz_2/conv1d_4/conv1d_1Conv2D7lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims:output:09lista__toeplitz_2/conv1d_4/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2%
#lista__toeplitz_2/conv1d_4/conv1d_1�
+lista__toeplitz_2/conv1d_4/conv1d_1/SqueezeSqueeze,lista__toeplitz_2/conv1d_4/conv1d_1:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2-
+lista__toeplitz_2/conv1d_4/conv1d_1/Squeeze�
2lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims/dim�
.lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims
ExpandDimslista__toeplitz_2/mul_2:z:0;lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������20
.lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims�
?lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpFlista__toeplitz_2_conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02A
?lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims_1/ReadVariableOp�
4lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims_1/dim�
0lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims_1
ExpandDimsGlista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims_1/ReadVariableOp:value:0=lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�22
0lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims_1�
#lista__toeplitz_2/conv1d_5/conv1d_1Conv2D7lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims:output:09lista__toeplitz_2/conv1d_5/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2%
#lista__toeplitz_2/conv1d_5/conv1d_1�
+lista__toeplitz_2/conv1d_5/conv1d_1/SqueezeSqueeze,lista__toeplitz_2/conv1d_5/conv1d_1:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2-
+lista__toeplitz_2/conv1d_5/conv1d_1/Squeeze�
lista__toeplitz_2/add_1AddV24lista__toeplitz_2/conv1d_4/conv1d_1/Squeeze:output:04lista__toeplitz_2/conv1d_5/conv1d_1/Squeeze:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/add_1�
,lista__toeplitz_2/Tensordot_4/ReadVariableOpReadVariableOp3lista__toeplitz_2_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,lista__toeplitz_2/Tensordot_4/ReadVariableOp�
"lista__toeplitz_2/Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"lista__toeplitz_2/Tensordot_4/axes�
"lista__toeplitz_2/Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"lista__toeplitz_2/Tensordot_4/free�
#lista__toeplitz_2/Tensordot_4/ShapeShapelista__toeplitz_2/Imag:output:0*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_4/Shape�
+lista__toeplitz_2/Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_4/GatherV2/axis�
&lista__toeplitz_2/Tensordot_4/GatherV2GatherV2,lista__toeplitz_2/Tensordot_4/Shape:output:0+lista__toeplitz_2/Tensordot_4/free:output:04lista__toeplitz_2/Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_4/GatherV2�
-lista__toeplitz_2/Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-lista__toeplitz_2/Tensordot_4/GatherV2_1/axis�
(lista__toeplitz_2/Tensordot_4/GatherV2_1GatherV2,lista__toeplitz_2/Tensordot_4/Shape:output:0+lista__toeplitz_2/Tensordot_4/axes:output:06lista__toeplitz_2/Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(lista__toeplitz_2/Tensordot_4/GatherV2_1�
#lista__toeplitz_2/Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#lista__toeplitz_2/Tensordot_4/Const�
"lista__toeplitz_2/Tensordot_4/ProdProd/lista__toeplitz_2/Tensordot_4/GatherV2:output:0,lista__toeplitz_2/Tensordot_4/Const:output:0*
T0*
_output_shapes
: 2$
"lista__toeplitz_2/Tensordot_4/Prod�
%lista__toeplitz_2/Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%lista__toeplitz_2/Tensordot_4/Const_1�
$lista__toeplitz_2/Tensordot_4/Prod_1Prod1lista__toeplitz_2/Tensordot_4/GatherV2_1:output:0.lista__toeplitz_2/Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: 2&
$lista__toeplitz_2/Tensordot_4/Prod_1�
)lista__toeplitz_2/Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot_4/concat/axis�
$lista__toeplitz_2/Tensordot_4/concatConcatV2+lista__toeplitz_2/Tensordot_4/axes:output:0+lista__toeplitz_2/Tensordot_4/free:output:02lista__toeplitz_2/Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot_4/concat�
#lista__toeplitz_2/Tensordot_4/stackPack-lista__toeplitz_2/Tensordot_4/Prod_1:output:0+lista__toeplitz_2/Tensordot_4/Prod:output:0*
N*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_4/stack�
'lista__toeplitz_2/Tensordot_4/transpose	Transposelista__toeplitz_2/Imag:output:0-lista__toeplitz_2/Tensordot_4/concat:output:0*
T0*+
_output_shapes
:@���������2)
'lista__toeplitz_2/Tensordot_4/transpose�
%lista__toeplitz_2/Tensordot_4/ReshapeReshape+lista__toeplitz_2/Tensordot_4/transpose:y:0,lista__toeplitz_2/Tensordot_4/stack:output:0*
T0*0
_output_shapes
:������������������2'
%lista__toeplitz_2/Tensordot_4/Reshape�
$lista__toeplitz_2/Tensordot_4/MatMulMatMul4lista__toeplitz_2/Tensordot_4/ReadVariableOp:value:0.lista__toeplitz_2/Tensordot_4/Reshape:output:0*
T0*(
_output_shapes
:����������2&
$lista__toeplitz_2/Tensordot_4/MatMul�
%lista__toeplitz_2/Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%lista__toeplitz_2/Tensordot_4/Const_2�
+lista__toeplitz_2/Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_4/concat_1/axis�
&lista__toeplitz_2/Tensordot_4/concat_1ConcatV2.lista__toeplitz_2/Tensordot_4/Const_2:output:0/lista__toeplitz_2/Tensordot_4/GatherV2:output:04lista__toeplitz_2/Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_4/concat_1�
lista__toeplitz_2/Tensordot_4Reshape.lista__toeplitz_2/Tensordot_4/MatMul:product:0/lista__toeplitz_2/Tensordot_4/concat_1:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tensordot_4�
"lista__toeplitz_2/transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"lista__toeplitz_2/transpose_4/perm�
lista__toeplitz_2/transpose_4	Transpose&lista__toeplitz_2/Tensordot_4:output:0+lista__toeplitz_2/transpose_4/perm:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/transpose_4�
lista__toeplitz_2/add_2AddV2lista__toeplitz_2/add_1:z:0!lista__toeplitz_2/transpose_4:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/add_2�
,lista__toeplitz_2/Tensordot_5/ReadVariableOpReadVariableOp5lista__toeplitz_2_tensordot_1_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,lista__toeplitz_2/Tensordot_5/ReadVariableOp�
"lista__toeplitz_2/Tensordot_5/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"lista__toeplitz_2/Tensordot_5/axes�
"lista__toeplitz_2/Tensordot_5/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"lista__toeplitz_2/Tensordot_5/free�
#lista__toeplitz_2/Tensordot_5/ShapeShapelista__toeplitz_2/Real:output:0*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_5/Shape�
+lista__toeplitz_2/Tensordot_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_5/GatherV2/axis�
&lista__toeplitz_2/Tensordot_5/GatherV2GatherV2,lista__toeplitz_2/Tensordot_5/Shape:output:0+lista__toeplitz_2/Tensordot_5/free:output:04lista__toeplitz_2/Tensordot_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_5/GatherV2�
-lista__toeplitz_2/Tensordot_5/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-lista__toeplitz_2/Tensordot_5/GatherV2_1/axis�
(lista__toeplitz_2/Tensordot_5/GatherV2_1GatherV2,lista__toeplitz_2/Tensordot_5/Shape:output:0+lista__toeplitz_2/Tensordot_5/axes:output:06lista__toeplitz_2/Tensordot_5/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(lista__toeplitz_2/Tensordot_5/GatherV2_1�
#lista__toeplitz_2/Tensordot_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#lista__toeplitz_2/Tensordot_5/Const�
"lista__toeplitz_2/Tensordot_5/ProdProd/lista__toeplitz_2/Tensordot_5/GatherV2:output:0,lista__toeplitz_2/Tensordot_5/Const:output:0*
T0*
_output_shapes
: 2$
"lista__toeplitz_2/Tensordot_5/Prod�
%lista__toeplitz_2/Tensordot_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%lista__toeplitz_2/Tensordot_5/Const_1�
$lista__toeplitz_2/Tensordot_5/Prod_1Prod1lista__toeplitz_2/Tensordot_5/GatherV2_1:output:0.lista__toeplitz_2/Tensordot_5/Const_1:output:0*
T0*
_output_shapes
: 2&
$lista__toeplitz_2/Tensordot_5/Prod_1�
)lista__toeplitz_2/Tensordot_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot_5/concat/axis�
$lista__toeplitz_2/Tensordot_5/concatConcatV2+lista__toeplitz_2/Tensordot_5/axes:output:0+lista__toeplitz_2/Tensordot_5/free:output:02lista__toeplitz_2/Tensordot_5/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot_5/concat�
#lista__toeplitz_2/Tensordot_5/stackPack-lista__toeplitz_2/Tensordot_5/Prod_1:output:0+lista__toeplitz_2/Tensordot_5/Prod:output:0*
N*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_5/stack�
'lista__toeplitz_2/Tensordot_5/transpose	Transposelista__toeplitz_2/Real:output:0-lista__toeplitz_2/Tensordot_5/concat:output:0*
T0*+
_output_shapes
:@���������2)
'lista__toeplitz_2/Tensordot_5/transpose�
%lista__toeplitz_2/Tensordot_5/ReshapeReshape+lista__toeplitz_2/Tensordot_5/transpose:y:0,lista__toeplitz_2/Tensordot_5/stack:output:0*
T0*0
_output_shapes
:������������������2'
%lista__toeplitz_2/Tensordot_5/Reshape�
$lista__toeplitz_2/Tensordot_5/MatMulMatMul4lista__toeplitz_2/Tensordot_5/ReadVariableOp:value:0.lista__toeplitz_2/Tensordot_5/Reshape:output:0*
T0*(
_output_shapes
:����������2&
$lista__toeplitz_2/Tensordot_5/MatMul�
%lista__toeplitz_2/Tensordot_5/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%lista__toeplitz_2/Tensordot_5/Const_2�
+lista__toeplitz_2/Tensordot_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_5/concat_1/axis�
&lista__toeplitz_2/Tensordot_5/concat_1ConcatV2.lista__toeplitz_2/Tensordot_5/Const_2:output:0/lista__toeplitz_2/Tensordot_5/GatherV2:output:04lista__toeplitz_2/Tensordot_5/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_5/concat_1�
lista__toeplitz_2/Tensordot_5Reshape.lista__toeplitz_2/Tensordot_5/MatMul:product:0/lista__toeplitz_2/Tensordot_5/concat_1:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tensordot_5�
"lista__toeplitz_2/transpose_5/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"lista__toeplitz_2/transpose_5/perm�
lista__toeplitz_2/transpose_5	Transpose&lista__toeplitz_2/Tensordot_5:output:0+lista__toeplitz_2/transpose_5/perm:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/transpose_5�
lista__toeplitz_2/add_3AddV2lista__toeplitz_2/add_2:z:0!lista__toeplitz_2/transpose_5:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/add_3�
lista__toeplitz_2/norm_3/mulMullista__toeplitz_2/add_3:z:0lista__toeplitz_2/add_3:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm_3/mul�
.lista__toeplitz_2/norm_3/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:20
.lista__toeplitz_2/norm_3/Sum/reduction_indices�
lista__toeplitz_2/norm_3/SumSum lista__toeplitz_2/norm_3/mul:z:07lista__toeplitz_2/norm_3/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2
lista__toeplitz_2/norm_3/Sum�
lista__toeplitz_2/norm_3/SqrtSqrt%lista__toeplitz_2/norm_3/Sum:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm_3/Sqrt�
 lista__toeplitz_2/norm_3/SqueezeSqueeze!lista__toeplitz_2/norm_3/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2"
 lista__toeplitz_2/norm_3/Squeeze�
"lista__toeplitz_2/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lista__toeplitz_2/ExpandDims_3/dim�
lista__toeplitz_2/ExpandDims_3
ExpandDims)lista__toeplitz_2/norm_3/Squeeze:output:0+lista__toeplitz_2/ExpandDims_3/dim:output:0*
T0*,
_output_shapes
:����������2 
lista__toeplitz_2/ExpandDims_3�
"lista__toeplitz_2/Tile_3/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2$
"lista__toeplitz_2/Tile_3/multiples�
lista__toeplitz_2/Tile_3Tile'lista__toeplitz_2/ExpandDims_3:output:0+lista__toeplitz_2/Tile_3/multiples:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tile_3�
*lista__toeplitz_2/Maximum_3/ReadVariableOpReadVariableOp3lista__toeplitz_2_maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02,
*lista__toeplitz_2/Maximum_3/ReadVariableOp�
lista__toeplitz_2/Maximum_3Maximum!lista__toeplitz_2/Tile_3:output:02lista__toeplitz_2/Maximum_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Maximum_3�
"lista__toeplitz_2/ReadVariableOp_3ReadVariableOp3lista__toeplitz_2_maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02$
"lista__toeplitz_2/ReadVariableOp_3�
lista__toeplitz_2/truediv_3RealDiv*lista__toeplitz_2/ReadVariableOp_3:value:0lista__toeplitz_2/Maximum_3:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/truediv_3{
lista__toeplitz_2/sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
lista__toeplitz_2/sub_5/x�
lista__toeplitz_2/sub_5Sub"lista__toeplitz_2/sub_5/x:output:0lista__toeplitz_2/truediv_3:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/sub_5�
lista__toeplitz_2/mul_3Mullista__toeplitz_2/sub_5:z:0lista__toeplitz_2/add_3:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/mul_3�
2lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims/dim�
.lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims
ExpandDimslista__toeplitz_2/mul_2:z:0;lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������20
.lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims�
?lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOpFlista__toeplitz_2_conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02A
?lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims_1/ReadVariableOp�
4lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims_1/dim�
0lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims_1
ExpandDimsGlista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims_1/ReadVariableOp:value:0=lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�22
0lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims_1�
#lista__toeplitz_2/conv1d_4/conv1d_2Conv2D7lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims:output:09lista__toeplitz_2/conv1d_4/conv1d_2/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2%
#lista__toeplitz_2/conv1d_4/conv1d_2�
+lista__toeplitz_2/conv1d_4/conv1d_2/SqueezeSqueeze,lista__toeplitz_2/conv1d_4/conv1d_2:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2-
+lista__toeplitz_2/conv1d_4/conv1d_2/Squeeze�
2lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims/dim�
.lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims
ExpandDimslista__toeplitz_2/mul_3:z:0;lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������20
.lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims�
?lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOpFlista__toeplitz_2_conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02A
?lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims_1/ReadVariableOp�
4lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims_1/dim�
0lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims_1
ExpandDimsGlista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims_1/ReadVariableOp:value:0=lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�22
0lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims_1�
#lista__toeplitz_2/conv1d_5/conv1d_2Conv2D7lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims:output:09lista__toeplitz_2/conv1d_5/conv1d_2/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2%
#lista__toeplitz_2/conv1d_5/conv1d_2�
+lista__toeplitz_2/conv1d_5/conv1d_2/SqueezeSqueeze,lista__toeplitz_2/conv1d_5/conv1d_2:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2-
+lista__toeplitz_2/conv1d_5/conv1d_2/Squeeze�
lista__toeplitz_2/sub_6Sub4lista__toeplitz_2/conv1d_4/conv1d_2/Squeeze:output:04lista__toeplitz_2/conv1d_5/conv1d_2/Squeeze:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/sub_6�
,lista__toeplitz_2/Tensordot_6/ReadVariableOpReadVariableOp3lista__toeplitz_2_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,lista__toeplitz_2/Tensordot_6/ReadVariableOp�
"lista__toeplitz_2/Tensordot_6/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"lista__toeplitz_2/Tensordot_6/axes�
"lista__toeplitz_2/Tensordot_6/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"lista__toeplitz_2/Tensordot_6/free�
#lista__toeplitz_2/Tensordot_6/ShapeShapelista__toeplitz_2/Real:output:0*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_6/Shape�
+lista__toeplitz_2/Tensordot_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_6/GatherV2/axis�
&lista__toeplitz_2/Tensordot_6/GatherV2GatherV2,lista__toeplitz_2/Tensordot_6/Shape:output:0+lista__toeplitz_2/Tensordot_6/free:output:04lista__toeplitz_2/Tensordot_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_6/GatherV2�
-lista__toeplitz_2/Tensordot_6/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-lista__toeplitz_2/Tensordot_6/GatherV2_1/axis�
(lista__toeplitz_2/Tensordot_6/GatherV2_1GatherV2,lista__toeplitz_2/Tensordot_6/Shape:output:0+lista__toeplitz_2/Tensordot_6/axes:output:06lista__toeplitz_2/Tensordot_6/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(lista__toeplitz_2/Tensordot_6/GatherV2_1�
#lista__toeplitz_2/Tensordot_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#lista__toeplitz_2/Tensordot_6/Const�
"lista__toeplitz_2/Tensordot_6/ProdProd/lista__toeplitz_2/Tensordot_6/GatherV2:output:0,lista__toeplitz_2/Tensordot_6/Const:output:0*
T0*
_output_shapes
: 2$
"lista__toeplitz_2/Tensordot_6/Prod�
%lista__toeplitz_2/Tensordot_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%lista__toeplitz_2/Tensordot_6/Const_1�
$lista__toeplitz_2/Tensordot_6/Prod_1Prod1lista__toeplitz_2/Tensordot_6/GatherV2_1:output:0.lista__toeplitz_2/Tensordot_6/Const_1:output:0*
T0*
_output_shapes
: 2&
$lista__toeplitz_2/Tensordot_6/Prod_1�
)lista__toeplitz_2/Tensordot_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot_6/concat/axis�
$lista__toeplitz_2/Tensordot_6/concatConcatV2+lista__toeplitz_2/Tensordot_6/axes:output:0+lista__toeplitz_2/Tensordot_6/free:output:02lista__toeplitz_2/Tensordot_6/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot_6/concat�
#lista__toeplitz_2/Tensordot_6/stackPack-lista__toeplitz_2/Tensordot_6/Prod_1:output:0+lista__toeplitz_2/Tensordot_6/Prod:output:0*
N*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_6/stack�
'lista__toeplitz_2/Tensordot_6/transpose	Transposelista__toeplitz_2/Real:output:0-lista__toeplitz_2/Tensordot_6/concat:output:0*
T0*+
_output_shapes
:@���������2)
'lista__toeplitz_2/Tensordot_6/transpose�
%lista__toeplitz_2/Tensordot_6/ReshapeReshape+lista__toeplitz_2/Tensordot_6/transpose:y:0,lista__toeplitz_2/Tensordot_6/stack:output:0*
T0*0
_output_shapes
:������������������2'
%lista__toeplitz_2/Tensordot_6/Reshape�
$lista__toeplitz_2/Tensordot_6/MatMulMatMul4lista__toeplitz_2/Tensordot_6/ReadVariableOp:value:0.lista__toeplitz_2/Tensordot_6/Reshape:output:0*
T0*(
_output_shapes
:����������2&
$lista__toeplitz_2/Tensordot_6/MatMul�
%lista__toeplitz_2/Tensordot_6/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%lista__toeplitz_2/Tensordot_6/Const_2�
+lista__toeplitz_2/Tensordot_6/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_6/concat_1/axis�
&lista__toeplitz_2/Tensordot_6/concat_1ConcatV2.lista__toeplitz_2/Tensordot_6/Const_2:output:0/lista__toeplitz_2/Tensordot_6/GatherV2:output:04lista__toeplitz_2/Tensordot_6/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_6/concat_1�
lista__toeplitz_2/Tensordot_6Reshape.lista__toeplitz_2/Tensordot_6/MatMul:product:0/lista__toeplitz_2/Tensordot_6/concat_1:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tensordot_6�
"lista__toeplitz_2/transpose_6/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"lista__toeplitz_2/transpose_6/perm�
lista__toeplitz_2/transpose_6	Transpose&lista__toeplitz_2/Tensordot_6:output:0+lista__toeplitz_2/transpose_6/perm:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/transpose_6�
lista__toeplitz_2/add_4AddV2lista__toeplitz_2/sub_6:z:0!lista__toeplitz_2/transpose_6:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/add_4�
,lista__toeplitz_2/Tensordot_7/ReadVariableOpReadVariableOp5lista__toeplitz_2_tensordot_1_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,lista__toeplitz_2/Tensordot_7/ReadVariableOp�
"lista__toeplitz_2/Tensordot_7/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"lista__toeplitz_2/Tensordot_7/axes�
"lista__toeplitz_2/Tensordot_7/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"lista__toeplitz_2/Tensordot_7/free�
#lista__toeplitz_2/Tensordot_7/ShapeShapelista__toeplitz_2/Imag:output:0*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_7/Shape�
+lista__toeplitz_2/Tensordot_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_7/GatherV2/axis�
&lista__toeplitz_2/Tensordot_7/GatherV2GatherV2,lista__toeplitz_2/Tensordot_7/Shape:output:0+lista__toeplitz_2/Tensordot_7/free:output:04lista__toeplitz_2/Tensordot_7/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_7/GatherV2�
-lista__toeplitz_2/Tensordot_7/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-lista__toeplitz_2/Tensordot_7/GatherV2_1/axis�
(lista__toeplitz_2/Tensordot_7/GatherV2_1GatherV2,lista__toeplitz_2/Tensordot_7/Shape:output:0+lista__toeplitz_2/Tensordot_7/axes:output:06lista__toeplitz_2/Tensordot_7/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(lista__toeplitz_2/Tensordot_7/GatherV2_1�
#lista__toeplitz_2/Tensordot_7/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#lista__toeplitz_2/Tensordot_7/Const�
"lista__toeplitz_2/Tensordot_7/ProdProd/lista__toeplitz_2/Tensordot_7/GatherV2:output:0,lista__toeplitz_2/Tensordot_7/Const:output:0*
T0*
_output_shapes
: 2$
"lista__toeplitz_2/Tensordot_7/Prod�
%lista__toeplitz_2/Tensordot_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%lista__toeplitz_2/Tensordot_7/Const_1�
$lista__toeplitz_2/Tensordot_7/Prod_1Prod1lista__toeplitz_2/Tensordot_7/GatherV2_1:output:0.lista__toeplitz_2/Tensordot_7/Const_1:output:0*
T0*
_output_shapes
: 2&
$lista__toeplitz_2/Tensordot_7/Prod_1�
)lista__toeplitz_2/Tensordot_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot_7/concat/axis�
$lista__toeplitz_2/Tensordot_7/concatConcatV2+lista__toeplitz_2/Tensordot_7/axes:output:0+lista__toeplitz_2/Tensordot_7/free:output:02lista__toeplitz_2/Tensordot_7/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot_7/concat�
#lista__toeplitz_2/Tensordot_7/stackPack-lista__toeplitz_2/Tensordot_7/Prod_1:output:0+lista__toeplitz_2/Tensordot_7/Prod:output:0*
N*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_7/stack�
'lista__toeplitz_2/Tensordot_7/transpose	Transposelista__toeplitz_2/Imag:output:0-lista__toeplitz_2/Tensordot_7/concat:output:0*
T0*+
_output_shapes
:@���������2)
'lista__toeplitz_2/Tensordot_7/transpose�
%lista__toeplitz_2/Tensordot_7/ReshapeReshape+lista__toeplitz_2/Tensordot_7/transpose:y:0,lista__toeplitz_2/Tensordot_7/stack:output:0*
T0*0
_output_shapes
:������������������2'
%lista__toeplitz_2/Tensordot_7/Reshape�
$lista__toeplitz_2/Tensordot_7/MatMulMatMul4lista__toeplitz_2/Tensordot_7/ReadVariableOp:value:0.lista__toeplitz_2/Tensordot_7/Reshape:output:0*
T0*(
_output_shapes
:����������2&
$lista__toeplitz_2/Tensordot_7/MatMul�
%lista__toeplitz_2/Tensordot_7/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%lista__toeplitz_2/Tensordot_7/Const_2�
+lista__toeplitz_2/Tensordot_7/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_7/concat_1/axis�
&lista__toeplitz_2/Tensordot_7/concat_1ConcatV2.lista__toeplitz_2/Tensordot_7/Const_2:output:0/lista__toeplitz_2/Tensordot_7/GatherV2:output:04lista__toeplitz_2/Tensordot_7/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_7/concat_1�
lista__toeplitz_2/Tensordot_7Reshape.lista__toeplitz_2/Tensordot_7/MatMul:product:0/lista__toeplitz_2/Tensordot_7/concat_1:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tensordot_7�
"lista__toeplitz_2/transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"lista__toeplitz_2/transpose_7/perm�
lista__toeplitz_2/transpose_7	Transpose&lista__toeplitz_2/Tensordot_7:output:0+lista__toeplitz_2/transpose_7/perm:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/transpose_7�
lista__toeplitz_2/sub_7Sublista__toeplitz_2/add_4:z:0!lista__toeplitz_2/transpose_7:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/sub_7�
lista__toeplitz_2/norm_4/mulMullista__toeplitz_2/sub_7:z:0lista__toeplitz_2/sub_7:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm_4/mul�
.lista__toeplitz_2/norm_4/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:20
.lista__toeplitz_2/norm_4/Sum/reduction_indices�
lista__toeplitz_2/norm_4/SumSum lista__toeplitz_2/norm_4/mul:z:07lista__toeplitz_2/norm_4/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2
lista__toeplitz_2/norm_4/Sum�
lista__toeplitz_2/norm_4/SqrtSqrt%lista__toeplitz_2/norm_4/Sum:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm_4/Sqrt�
 lista__toeplitz_2/norm_4/SqueezeSqueeze!lista__toeplitz_2/norm_4/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2"
 lista__toeplitz_2/norm_4/Squeeze�
"lista__toeplitz_2/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lista__toeplitz_2/ExpandDims_4/dim�
lista__toeplitz_2/ExpandDims_4
ExpandDims)lista__toeplitz_2/norm_4/Squeeze:output:0+lista__toeplitz_2/ExpandDims_4/dim:output:0*
T0*,
_output_shapes
:����������2 
lista__toeplitz_2/ExpandDims_4�
"lista__toeplitz_2/Tile_4/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2$
"lista__toeplitz_2/Tile_4/multiples�
lista__toeplitz_2/Tile_4Tile'lista__toeplitz_2/ExpandDims_4:output:0+lista__toeplitz_2/Tile_4/multiples:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tile_4�
*lista__toeplitz_2/Maximum_4/ReadVariableOpReadVariableOp3lista__toeplitz_2_maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02,
*lista__toeplitz_2/Maximum_4/ReadVariableOp�
lista__toeplitz_2/Maximum_4Maximum!lista__toeplitz_2/Tile_4:output:02lista__toeplitz_2/Maximum_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Maximum_4�
"lista__toeplitz_2/ReadVariableOp_4ReadVariableOp3lista__toeplitz_2_maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02$
"lista__toeplitz_2/ReadVariableOp_4�
lista__toeplitz_2/truediv_4RealDiv*lista__toeplitz_2/ReadVariableOp_4:value:0lista__toeplitz_2/Maximum_4:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/truediv_4{
lista__toeplitz_2/sub_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
lista__toeplitz_2/sub_8/x�
lista__toeplitz_2/sub_8Sub"lista__toeplitz_2/sub_8/x:output:0lista__toeplitz_2/truediv_4:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/sub_8�
lista__toeplitz_2/mul_4Mullista__toeplitz_2/sub_8:z:0lista__toeplitz_2/sub_7:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/mul_4�
2lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims/dim�
.lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims
ExpandDimslista__toeplitz_2/mul_3:z:0;lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������20
.lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims�
?lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOpFlista__toeplitz_2_conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02A
?lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims_1/ReadVariableOp�
4lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims_1/dim�
0lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims_1
ExpandDimsGlista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims_1/ReadVariableOp:value:0=lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�22
0lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims_1�
#lista__toeplitz_2/conv1d_4/conv1d_3Conv2D7lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims:output:09lista__toeplitz_2/conv1d_4/conv1d_3/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2%
#lista__toeplitz_2/conv1d_4/conv1d_3�
+lista__toeplitz_2/conv1d_4/conv1d_3/SqueezeSqueeze,lista__toeplitz_2/conv1d_4/conv1d_3:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2-
+lista__toeplitz_2/conv1d_4/conv1d_3/Squeeze�
2lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims/dim�
.lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims
ExpandDimslista__toeplitz_2/mul_4:z:0;lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������20
.lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims�
?lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOpFlista__toeplitz_2_conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype02A
?lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims_1/ReadVariableOp�
4lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims_1/dim�
0lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims_1
ExpandDimsGlista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims_1/ReadVariableOp:value:0=lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�22
0lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims_1�
#lista__toeplitz_2/conv1d_5/conv1d_3Conv2D7lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims:output:09lista__toeplitz_2/conv1d_5/conv1d_3/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2%
#lista__toeplitz_2/conv1d_5/conv1d_3�
+lista__toeplitz_2/conv1d_5/conv1d_3/SqueezeSqueeze,lista__toeplitz_2/conv1d_5/conv1d_3:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2-
+lista__toeplitz_2/conv1d_5/conv1d_3/Squeeze�
lista__toeplitz_2/add_5AddV24lista__toeplitz_2/conv1d_4/conv1d_3/Squeeze:output:04lista__toeplitz_2/conv1d_5/conv1d_3/Squeeze:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/add_5�
,lista__toeplitz_2/Tensordot_8/ReadVariableOpReadVariableOp3lista__toeplitz_2_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,lista__toeplitz_2/Tensordot_8/ReadVariableOp�
"lista__toeplitz_2/Tensordot_8/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"lista__toeplitz_2/Tensordot_8/axes�
"lista__toeplitz_2/Tensordot_8/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"lista__toeplitz_2/Tensordot_8/free�
#lista__toeplitz_2/Tensordot_8/ShapeShapelista__toeplitz_2/Imag:output:0*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_8/Shape�
+lista__toeplitz_2/Tensordot_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_8/GatherV2/axis�
&lista__toeplitz_2/Tensordot_8/GatherV2GatherV2,lista__toeplitz_2/Tensordot_8/Shape:output:0+lista__toeplitz_2/Tensordot_8/free:output:04lista__toeplitz_2/Tensordot_8/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_8/GatherV2�
-lista__toeplitz_2/Tensordot_8/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-lista__toeplitz_2/Tensordot_8/GatherV2_1/axis�
(lista__toeplitz_2/Tensordot_8/GatherV2_1GatherV2,lista__toeplitz_2/Tensordot_8/Shape:output:0+lista__toeplitz_2/Tensordot_8/axes:output:06lista__toeplitz_2/Tensordot_8/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(lista__toeplitz_2/Tensordot_8/GatherV2_1�
#lista__toeplitz_2/Tensordot_8/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#lista__toeplitz_2/Tensordot_8/Const�
"lista__toeplitz_2/Tensordot_8/ProdProd/lista__toeplitz_2/Tensordot_8/GatherV2:output:0,lista__toeplitz_2/Tensordot_8/Const:output:0*
T0*
_output_shapes
: 2$
"lista__toeplitz_2/Tensordot_8/Prod�
%lista__toeplitz_2/Tensordot_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%lista__toeplitz_2/Tensordot_8/Const_1�
$lista__toeplitz_2/Tensordot_8/Prod_1Prod1lista__toeplitz_2/Tensordot_8/GatherV2_1:output:0.lista__toeplitz_2/Tensordot_8/Const_1:output:0*
T0*
_output_shapes
: 2&
$lista__toeplitz_2/Tensordot_8/Prod_1�
)lista__toeplitz_2/Tensordot_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot_8/concat/axis�
$lista__toeplitz_2/Tensordot_8/concatConcatV2+lista__toeplitz_2/Tensordot_8/axes:output:0+lista__toeplitz_2/Tensordot_8/free:output:02lista__toeplitz_2/Tensordot_8/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot_8/concat�
#lista__toeplitz_2/Tensordot_8/stackPack-lista__toeplitz_2/Tensordot_8/Prod_1:output:0+lista__toeplitz_2/Tensordot_8/Prod:output:0*
N*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_8/stack�
'lista__toeplitz_2/Tensordot_8/transpose	Transposelista__toeplitz_2/Imag:output:0-lista__toeplitz_2/Tensordot_8/concat:output:0*
T0*+
_output_shapes
:@���������2)
'lista__toeplitz_2/Tensordot_8/transpose�
%lista__toeplitz_2/Tensordot_8/ReshapeReshape+lista__toeplitz_2/Tensordot_8/transpose:y:0,lista__toeplitz_2/Tensordot_8/stack:output:0*
T0*0
_output_shapes
:������������������2'
%lista__toeplitz_2/Tensordot_8/Reshape�
$lista__toeplitz_2/Tensordot_8/MatMulMatMul4lista__toeplitz_2/Tensordot_8/ReadVariableOp:value:0.lista__toeplitz_2/Tensordot_8/Reshape:output:0*
T0*(
_output_shapes
:����������2&
$lista__toeplitz_2/Tensordot_8/MatMul�
%lista__toeplitz_2/Tensordot_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%lista__toeplitz_2/Tensordot_8/Const_2�
+lista__toeplitz_2/Tensordot_8/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_8/concat_1/axis�
&lista__toeplitz_2/Tensordot_8/concat_1ConcatV2.lista__toeplitz_2/Tensordot_8/Const_2:output:0/lista__toeplitz_2/Tensordot_8/GatherV2:output:04lista__toeplitz_2/Tensordot_8/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_8/concat_1�
lista__toeplitz_2/Tensordot_8Reshape.lista__toeplitz_2/Tensordot_8/MatMul:product:0/lista__toeplitz_2/Tensordot_8/concat_1:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tensordot_8�
"lista__toeplitz_2/transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"lista__toeplitz_2/transpose_8/perm�
lista__toeplitz_2/transpose_8	Transpose&lista__toeplitz_2/Tensordot_8:output:0+lista__toeplitz_2/transpose_8/perm:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/transpose_8�
lista__toeplitz_2/add_6AddV2lista__toeplitz_2/add_5:z:0!lista__toeplitz_2/transpose_8:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/add_6�
,lista__toeplitz_2/Tensordot_9/ReadVariableOpReadVariableOp5lista__toeplitz_2_tensordot_1_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,lista__toeplitz_2/Tensordot_9/ReadVariableOp�
"lista__toeplitz_2/Tensordot_9/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"lista__toeplitz_2/Tensordot_9/axes�
"lista__toeplitz_2/Tensordot_9/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"lista__toeplitz_2/Tensordot_9/free�
#lista__toeplitz_2/Tensordot_9/ShapeShapelista__toeplitz_2/Real:output:0*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_9/Shape�
+lista__toeplitz_2/Tensordot_9/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_9/GatherV2/axis�
&lista__toeplitz_2/Tensordot_9/GatherV2GatherV2,lista__toeplitz_2/Tensordot_9/Shape:output:0+lista__toeplitz_2/Tensordot_9/free:output:04lista__toeplitz_2/Tensordot_9/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_9/GatherV2�
-lista__toeplitz_2/Tensordot_9/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-lista__toeplitz_2/Tensordot_9/GatherV2_1/axis�
(lista__toeplitz_2/Tensordot_9/GatherV2_1GatherV2,lista__toeplitz_2/Tensordot_9/Shape:output:0+lista__toeplitz_2/Tensordot_9/axes:output:06lista__toeplitz_2/Tensordot_9/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(lista__toeplitz_2/Tensordot_9/GatherV2_1�
#lista__toeplitz_2/Tensordot_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#lista__toeplitz_2/Tensordot_9/Const�
"lista__toeplitz_2/Tensordot_9/ProdProd/lista__toeplitz_2/Tensordot_9/GatherV2:output:0,lista__toeplitz_2/Tensordot_9/Const:output:0*
T0*
_output_shapes
: 2$
"lista__toeplitz_2/Tensordot_9/Prod�
%lista__toeplitz_2/Tensordot_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%lista__toeplitz_2/Tensordot_9/Const_1�
$lista__toeplitz_2/Tensordot_9/Prod_1Prod1lista__toeplitz_2/Tensordot_9/GatherV2_1:output:0.lista__toeplitz_2/Tensordot_9/Const_1:output:0*
T0*
_output_shapes
: 2&
$lista__toeplitz_2/Tensordot_9/Prod_1�
)lista__toeplitz_2/Tensordot_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)lista__toeplitz_2/Tensordot_9/concat/axis�
$lista__toeplitz_2/Tensordot_9/concatConcatV2+lista__toeplitz_2/Tensordot_9/axes:output:0+lista__toeplitz_2/Tensordot_9/free:output:02lista__toeplitz_2/Tensordot_9/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$lista__toeplitz_2/Tensordot_9/concat�
#lista__toeplitz_2/Tensordot_9/stackPack-lista__toeplitz_2/Tensordot_9/Prod_1:output:0+lista__toeplitz_2/Tensordot_9/Prod:output:0*
N*
T0*
_output_shapes
:2%
#lista__toeplitz_2/Tensordot_9/stack�
'lista__toeplitz_2/Tensordot_9/transpose	Transposelista__toeplitz_2/Real:output:0-lista__toeplitz_2/Tensordot_9/concat:output:0*
T0*+
_output_shapes
:@���������2)
'lista__toeplitz_2/Tensordot_9/transpose�
%lista__toeplitz_2/Tensordot_9/ReshapeReshape+lista__toeplitz_2/Tensordot_9/transpose:y:0,lista__toeplitz_2/Tensordot_9/stack:output:0*
T0*0
_output_shapes
:������������������2'
%lista__toeplitz_2/Tensordot_9/Reshape�
$lista__toeplitz_2/Tensordot_9/MatMulMatMul4lista__toeplitz_2/Tensordot_9/ReadVariableOp:value:0.lista__toeplitz_2/Tensordot_9/Reshape:output:0*
T0*(
_output_shapes
:����������2&
$lista__toeplitz_2/Tensordot_9/MatMul�
%lista__toeplitz_2/Tensordot_9/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%lista__toeplitz_2/Tensordot_9/Const_2�
+lista__toeplitz_2/Tensordot_9/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lista__toeplitz_2/Tensordot_9/concat_1/axis�
&lista__toeplitz_2/Tensordot_9/concat_1ConcatV2.lista__toeplitz_2/Tensordot_9/Const_2:output:0/lista__toeplitz_2/Tensordot_9/GatherV2:output:04lista__toeplitz_2/Tensordot_9/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&lista__toeplitz_2/Tensordot_9/concat_1�
lista__toeplitz_2/Tensordot_9Reshape.lista__toeplitz_2/Tensordot_9/MatMul:product:0/lista__toeplitz_2/Tensordot_9/concat_1:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tensordot_9�
"lista__toeplitz_2/transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"lista__toeplitz_2/transpose_9/perm�
lista__toeplitz_2/transpose_9	Transpose&lista__toeplitz_2/Tensordot_9:output:0+lista__toeplitz_2/transpose_9/perm:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/transpose_9�
lista__toeplitz_2/add_7AddV2lista__toeplitz_2/add_6:z:0!lista__toeplitz_2/transpose_9:y:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/add_7�
lista__toeplitz_2/norm_5/mulMullista__toeplitz_2/add_7:z:0lista__toeplitz_2/add_7:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm_5/mul�
.lista__toeplitz_2/norm_5/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:20
.lista__toeplitz_2/norm_5/Sum/reduction_indices�
lista__toeplitz_2/norm_5/SumSum lista__toeplitz_2/norm_5/mul:z:07lista__toeplitz_2/norm_5/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2
lista__toeplitz_2/norm_5/Sum�
lista__toeplitz_2/norm_5/SqrtSqrt%lista__toeplitz_2/norm_5/Sum:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/norm_5/Sqrt�
 lista__toeplitz_2/norm_5/SqueezeSqueeze!lista__toeplitz_2/norm_5/Sqrt:y:0*
T0*(
_output_shapes
:����������*
squeeze_dims
2"
 lista__toeplitz_2/norm_5/Squeeze�
"lista__toeplitz_2/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lista__toeplitz_2/ExpandDims_5/dim�
lista__toeplitz_2/ExpandDims_5
ExpandDims)lista__toeplitz_2/norm_5/Squeeze:output:0+lista__toeplitz_2/ExpandDims_5/dim:output:0*
T0*,
_output_shapes
:����������2 
lista__toeplitz_2/ExpandDims_5�
"lista__toeplitz_2/Tile_5/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2$
"lista__toeplitz_2/Tile_5/multiples�
lista__toeplitz_2/Tile_5Tile'lista__toeplitz_2/ExpandDims_5:output:0+lista__toeplitz_2/Tile_5/multiples:output:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Tile_5�
*lista__toeplitz_2/Maximum_5/ReadVariableOpReadVariableOp3lista__toeplitz_2_maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02,
*lista__toeplitz_2/Maximum_5/ReadVariableOp�
lista__toeplitz_2/Maximum_5Maximum!lista__toeplitz_2/Tile_5:output:02lista__toeplitz_2/Maximum_5/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/Maximum_5�
"lista__toeplitz_2/ReadVariableOp_5ReadVariableOp3lista__toeplitz_2_maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02$
"lista__toeplitz_2/ReadVariableOp_5�
lista__toeplitz_2/truediv_5RealDiv*lista__toeplitz_2/ReadVariableOp_5:value:0lista__toeplitz_2/Maximum_5:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/truediv_5{
lista__toeplitz_2/sub_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
lista__toeplitz_2/sub_9/x�
lista__toeplitz_2/sub_9Sub"lista__toeplitz_2/sub_9/x:output:0lista__toeplitz_2/truediv_5:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/sub_9�
lista__toeplitz_2/mul_5Mullista__toeplitz_2/sub_9:z:0lista__toeplitz_2/add_7:z:0*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/mul_5�
lista__toeplitz_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
lista__toeplitz_2/concat/axis�
lista__toeplitz_2/concatConcatV2lista__toeplitz_2/mul_4:z:0lista__toeplitz_2/mul_5:z:0&lista__toeplitz_2/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������2
lista__toeplitz_2/concatz
IdentityIdentity!lista__toeplitz_2/concat:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������@::::::::T P
+
_output_shapes
:���������@
!
_user_specified_name	input_1
�x
�
"__inference__traced_restore_680624
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_28
4assignvariableop_3_lista__toeplitz_2_conv1d_4_kernel8
4assignvariableop_4_lista__toeplitz_2_conv1d_5_kernel!
assignvariableop_5_variable_3!
assignvariableop_6_variable_4!
assignvariableop_7_variable_5
assignvariableop_8_beta_1
assignvariableop_9_beta_2
assignvariableop_10_decay%
!assignvariableop_11_learning_rate!
assignvariableop_12_adam_iter
assignvariableop_13_total
assignvariableop_14_count'
#assignvariableop_15_adam_variable_m)
%assignvariableop_16_adam_variable_m_1@
<assignvariableop_17_adam_lista__toeplitz_2_conv1d_4_kernel_m@
<assignvariableop_18_adam_lista__toeplitz_2_conv1d_5_kernel_m)
%assignvariableop_19_adam_variable_m_2)
%assignvariableop_20_adam_variable_m_3)
%assignvariableop_21_adam_variable_m_4'
#assignvariableop_22_adam_variable_v)
%assignvariableop_23_adam_variable_v_1@
<assignvariableop_24_adam_lista__toeplitz_2_conv1d_4_kernel_v@
<assignvariableop_25_adam_lista__toeplitz_2_conv1d_5_kernel_v)
%assignvariableop_26_adam_variable_v_2)
%assignvariableop_27_adam_variable_v_3)
%assignvariableop_28_adam_variable_v_4
identity_30��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�BWe_r/.ATTRIBUTES/VARIABLE_VALUEBWe_i/.ATTRIBUTES/VARIABLE_VALUEB alpha/.ATTRIBUTES/VARIABLE_VALUEB&hg_r/kernel/.ATTRIBUTES/VARIABLE_VALUEB&hg_i/kernel/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/0/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/1/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/2/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB;We_r/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;We_i/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBhg_r/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBhg_i/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;We_r/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB;We_i/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBhg_r/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBhg_i/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp4assignvariableop_3_lista__toeplitz_2_conv1d_4_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp4assignvariableop_4_lista__toeplitz_2_conv1d_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_3Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_4Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_5Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_adam_variable_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_variable_m_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp<assignvariableop_17_adam_lista__toeplitz_2_conv1d_4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp<assignvariableop_18_adam_lista__toeplitz_2_conv1d_5_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp%assignvariableop_19_adam_variable_m_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_variable_m_3Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_variable_m_4Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp#assignvariableop_22_adam_variable_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp%assignvariableop_23_adam_variable_v_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp<assignvariableop_24_adam_lista__toeplitz_2_conv1d_4_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp<assignvariableop_25_adam_lista__toeplitz_2_conv1d_5_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_variable_v_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_variable_v_3Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_variable_v_4Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29�
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*�
_input_shapesx
v: :::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������@A
output_15
StatefulPartitionedCall:0����������tensorflow/serving/predict:�M
�
hg_r
hg_i
We_r
We_i
lam_list
	alpha
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
A__call__
*B&call_and_return_all_conditional_losses
C_default_save_signature"�
_tf_keras_model�{"class_name": "LISTA_Toeplitz", "name": "lista__toeplitz_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "LISTA_Toeplitz"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-08, "amsgrad": false}}}}
�	

kernel
	variables
regularization_losses
trainable_variables
	keras_api
D__call__
*E&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [359]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 180, 1]}}
�	

kernel
	variables
regularization_losses
trainable_variables
	keras_api
F__call__
*G&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [359]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 180, 1]}}
:	�@2Variable
:	�@2Variable
5
0
1
2"
trackable_list_wrapper
: 2Variable
�

beta_1

beta_2
	decay
learning_rate
iterm3m4m5m6m7m8m9v:v;v<v=v>v?v@"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
�
	variables
non_trainable_variables

 layers
!metrics
	regularization_losses

trainable_variables
"layer_metrics
#layer_regularization_losses
A__call__
C_default_save_signature
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
,
Hserving_default"
signature_map
8:6�2!lista__toeplitz_2/conv1d_4/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
	variables
$non_trainable_variables
%metrics

&layers
regularization_losses
trainable_variables
'layer_metrics
(layer_regularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
8:6�2!lista__toeplitz_2/conv1d_5/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
	variables
)non_trainable_variables
*metrics

+layers
regularization_losses
trainable_variables
,layer_metrics
-layer_regularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
: 2Variable
: 2Variable
: 2Variable
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
.0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
	/total
	0count
1	variables
2	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
/0
01"
trackable_list_wrapper
-
1	variables"
_generic_user_object
 :	�@2Adam/Variable/m
 :	�@2Adam/Variable/m
=:;�2(Adam/lista__toeplitz_2/conv1d_4/kernel/m
=:;�2(Adam/lista__toeplitz_2/conv1d_5/kernel/m
: 2Adam/Variable/m
: 2Adam/Variable/m
: 2Adam/Variable/m
 :	�@2Adam/Variable/v
 :	�@2Adam/Variable/v
=:;�2(Adam/lista__toeplitz_2/conv1d_4/kernel/v
=:;�2(Adam/lista__toeplitz_2/conv1d_5/kernel/v
: 2Adam/Variable/v
: 2Adam/Variable/v
: 2Adam/Variable/v
�2�
2__inference_lista__toeplitz_2_layer_call_fn_680350�
���
FullArgSpec
args�
jself
jY
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������@
�2�
M__inference_lista__toeplitz_2_layer_call_and_return_conditional_losses_680330�
���
FullArgSpec
args�
jself
jY
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������@
�2�
!__inference__wrapped_model_679908�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������@
�2�
)__inference_conv1d_4_layer_call_fn_680398�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv1d_4_layer_call_and_return_conditional_losses_680391�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv1d_5_layer_call_fn_680417�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv1d_5_layer_call_and_return_conditional_losses_680410�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
3B1
$__inference_signature_wrapper_680379input_1�
!__inference__wrapped_model_679908y4�1
*�'
%�"
input_1���������@
� "8�5
3
output_1'�$
output_1�����������
D__inference_conv1d_4_layer_call_and_return_conditional_losses_680391e4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
)__inference_conv1d_4_layer_call_fn_680398X4�1
*�'
%�"
inputs����������
� "������������
D__inference_conv1d_5_layer_call_and_return_conditional_losses_680410e4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
)__inference_conv1d_5_layer_call_fn_680417X4�1
*�'
%�"
inputs����������
� "������������
M__inference_lista__toeplitz_2_layer_call_and_return_conditional_losses_680330k4�1
*�'
%�"
input_1���������@
� "*�'
 �
0����������
� �
2__inference_lista__toeplitz_2_layer_call_fn_680350^4�1
*�'
%�"
input_1���������@
� "������������
$__inference_signature_wrapper_680379�?�<
� 
5�2
0
input_1%�"
input_1���������@"8�5
3
output_1'�$
output_1����������