Ѓ
ПЃ
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
dtypetype
О
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12unknown8го
n
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДД*
shared_name
Variable
g
Variable/Read/ReadVariableOpReadVariableOpVariable* 
_output_shapes
:
ДД*
dtype0
r

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДД*
shared_name
Variable_1
k
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1* 
_output_shapes
:
ДД*
dtype0
q

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:	Д@*
shared_name
Variable_2
j
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:	Д@*
dtype0
q

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	Д@*
shared_name
Variable_3
j
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:	Д@*
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
h

Variable_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_6
a
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
: *
dtype0
h

Variable_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_7
a
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
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
|
Adam/Variable/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДД* 
shared_nameAdam/Variable/m
u
#Adam/Variable/m/Read/ReadVariableOpReadVariableOpAdam/Variable/m* 
_output_shapes
:
ДД*
dtype0

Adam/Variable/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДД*"
shared_nameAdam/Variable/m_1
y
%Adam/Variable/m_1/Read/ReadVariableOpReadVariableOpAdam/Variable/m_1* 
_output_shapes
:
ДД*
dtype0

Adam/Variable/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:	Д@*"
shared_nameAdam/Variable/m_2
x
%Adam/Variable/m_2/Read/ReadVariableOpReadVariableOpAdam/Variable/m_2*
_output_shapes
:	Д@*
dtype0

Adam/Variable/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	Д@*"
shared_nameAdam/Variable/m_3
x
%Adam/Variable/m_3/Read/ReadVariableOpReadVariableOpAdam/Variable/m_3*
_output_shapes
:	Д@*
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
v
Adam/Variable/m_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/m_5
o
%Adam/Variable/m_5/Read/ReadVariableOpReadVariableOpAdam/Variable/m_5*
_output_shapes
: *
dtype0
v
Adam/Variable/m_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/m_6
o
%Adam/Variable/m_6/Read/ReadVariableOpReadVariableOpAdam/Variable/m_6*
_output_shapes
: *
dtype0
|
Adam/Variable/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДД* 
shared_nameAdam/Variable/v
u
#Adam/Variable/v/Read/ReadVariableOpReadVariableOpAdam/Variable/v* 
_output_shapes
:
ДД*
dtype0

Adam/Variable/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДД*"
shared_nameAdam/Variable/v_1
y
%Adam/Variable/v_1/Read/ReadVariableOpReadVariableOpAdam/Variable/v_1* 
_output_shapes
:
ДД*
dtype0

Adam/Variable/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:	Д@*"
shared_nameAdam/Variable/v_2
x
%Adam/Variable/v_2/Read/ReadVariableOpReadVariableOpAdam/Variable/v_2*
_output_shapes
:	Д@*
dtype0

Adam/Variable/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	Д@*"
shared_nameAdam/Variable/v_3
x
%Adam/Variable/v_3/Read/ReadVariableOpReadVariableOpAdam/Variable/v_3*
_output_shapes
:	Д@*
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
v
Adam/Variable/v_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/v_5
o
%Adam/Variable/v_5/Read/ReadVariableOpReadVariableOpAdam/Variable/v_5*
_output_shapes
: *
dtype0
v
Adam/Variable/v_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/v_6
o
%Adam/Variable/v_6/Read/ReadVariableOpReadVariableOpAdam/Variable/v_6*
_output_shapes
: *
dtype0

NoOpNoOp
У
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ў
valueєBё Bъ
В
Wt_r
Wt_i
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
=;
VARIABLE_VALUEVariableWt_r/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
Variable_1Wt_i/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
Variable_2We_r/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
Variable_3We_i/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
@>
VARIABLE_VALUE
Variable_4 alpha/.ATTRIBUTES/VARIABLE_VALUE
О

beta_1

beta_2
	decay
learning_rate
itermm m!m"m#m$m%v&v'v(v)v*v+v,
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
­
	variables
non_trainable_variables

layers
metrics
	regularization_losses

trainable_variables
layer_metrics
layer_regularization_losses
 
EC
VARIABLE_VALUE
Variable_5%lam_list/0/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE
Variable_6%lam_list/1/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE
Variable_7%lam_list/2/.ATTRIBUTES/VARIABLE_VALUE
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
 

0
 
 
4
	total
	count
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
`^
VARIABLE_VALUEAdam/Variable/m;Wt_r/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEAdam/Variable/m_1;Wt_i/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEAdam/Variable/m_2;We_r/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEAdam/Variable/m_3;We_i/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/m_4Alam_list/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/m_5Alam_list/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/m_6Alam_list/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEAdam/Variable/v;Wt_r/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEAdam/Variable/v_1;Wt_i/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEAdam/Variable/v_2;We_r/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEAdam/Variable/v_3;We_i/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/v_4Alam_list/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/v_5Alam_list/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/Variable/v_6Alam_list/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*+
_output_shapes
:џџџџџџџџџ@*
dtype0* 
shape:џџџџџџџџџ@

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1
Variable_2
Variable_5
Variable_3Variable
Variable_1
Variable_6
Variable_7*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџш*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_903618
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ђ

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_4/Read/ReadVariableOpVariable_5/Read/ReadVariableOpVariable_6/Read/ReadVariableOpVariable_7/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp#Adam/Variable/m/Read/ReadVariableOp%Adam/Variable/m_1/Read/ReadVariableOp%Adam/Variable/m_2/Read/ReadVariableOp%Adam/Variable/m_3/Read/ReadVariableOp%Adam/Variable/m_4/Read/ReadVariableOp%Adam/Variable/m_5/Read/ReadVariableOp%Adam/Variable/m_6/Read/ReadVariableOp#Adam/Variable/v/Read/ReadVariableOp%Adam/Variable/v_1/Read/ReadVariableOp%Adam/Variable/v_2/Read/ReadVariableOp%Adam/Variable/v_3/Read/ReadVariableOp%Adam/Variable/v_4/Read/ReadVariableOp%Adam/Variable/v_5/Read/ReadVariableOp%Adam/Variable/v_6/Read/ReadVariableOpConst**
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_903728
й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1
Variable_2
Variable_3
Variable_4
Variable_5
Variable_6
Variable_7beta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/Variable/mAdam/Variable/m_1Adam/Variable/m_2Adam/Variable/m_3Adam/Variable/m_4Adam/Variable/m_5Adam/Variable/m_6Adam/Variable/vAdam/Variable/v_1Adam/Variable/v_2Adam/Variable/v_3Adam/Variable/v_4Adam/Variable/v_5Adam/Variable/v_6*)
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_903825Хќ
<

__inference__traced_save_903728
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_4_read_readvariableop)
%savev2_variable_5_read_readvariableop)
%savev2_variable_6_read_readvariableop)
%savev2_variable_7_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop.
*savev2_adam_variable_m_read_readvariableop0
,savev2_adam_variable_m_1_read_readvariableop0
,savev2_adam_variable_m_2_read_readvariableop0
,savev2_adam_variable_m_3_read_readvariableop0
,savev2_adam_variable_m_4_read_readvariableop0
,savev2_adam_variable_m_5_read_readvariableop0
,savev2_adam_variable_m_6_read_readvariableop.
*savev2_adam_variable_v_read_readvariableop0
,savev2_adam_variable_v_1_read_readvariableop0
,savev2_adam_variable_v_2_read_readvariableop0
,savev2_adam_variable_v_3_read_readvariableop0
,savev2_adam_variable_v_4_read_readvariableop0
,savev2_adam_variable_v_5_read_readvariableop0
,savev2_adam_variable_v_6_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_dba6b0bec3d84c9c8dcdae6d2862c6e1/part2	
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBBWt_r/.ATTRIBUTES/VARIABLE_VALUEBWt_i/.ATTRIBUTES/VARIABLE_VALUEBWe_r/.ATTRIBUTES/VARIABLE_VALUEBWe_i/.ATTRIBUTES/VARIABLE_VALUEB alpha/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/0/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/1/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/2/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB;Wt_r/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;Wt_i/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;We_r/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;We_i/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;Wt_r/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB;Wt_i/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB;We_r/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB;We_i/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesФ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_5_read_readvariableop%savev2_variable_6_read_readvariableop%savev2_variable_7_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop*savev2_adam_variable_m_read_readvariableop,savev2_adam_variable_m_1_read_readvariableop,savev2_adam_variable_m_2_read_readvariableop,savev2_adam_variable_m_3_read_readvariableop,savev2_adam_variable_m_4_read_readvariableop,savev2_adam_variable_m_5_read_readvariableop,savev2_adam_variable_m_6_read_readvariableop*savev2_adam_variable_v_read_readvariableop,savev2_adam_variable_v_1_read_readvariableop,savev2_adam_variable_v_2_read_readvariableop,savev2_adam_variable_v_3_read_readvariableop,savev2_adam_variable_v_4_read_readvariableop,savev2_adam_variable_v_5_read_readvariableop,savev2_adam_variable_v_6_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*Х
_input_shapesГ
А: :
ДД:
ДД:	Д@:	Д@: : : : : : : : : : : :
ДД:
ДД:	Д@:	Д@: : : :
ДД:
ДД:	Д@:	Д@: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ДД:&"
 
_output_shapes
:
ДД:%!

_output_shapes
:	Д@:%!

_output_shapes
:	Д@:

_output_shapes
: :
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
: :&"
 
_output_shapes
:
ДД:&"
 
_output_shapes
:
ДД:%!

_output_shapes
:	Д@:%!

_output_shapes
:	Д@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
ДД:&"
 
_output_shapes
:
ДД:%!

_output_shapes
:	Д@:%!

_output_shapes
:	Д@:
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
Пv

"__inference__traced_restore_903825
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_2!
assignvariableop_3_variable_3!
assignvariableop_4_variable_4!
assignvariableop_5_variable_5!
assignvariableop_6_variable_6!
assignvariableop_7_variable_7
assignvariableop_8_beta_1
assignvariableop_9_beta_2
assignvariableop_10_decay%
!assignvariableop_11_learning_rate!
assignvariableop_12_adam_iter
assignvariableop_13_total
assignvariableop_14_count'
#assignvariableop_15_adam_variable_m)
%assignvariableop_16_adam_variable_m_1)
%assignvariableop_17_adam_variable_m_2)
%assignvariableop_18_adam_variable_m_3)
%assignvariableop_19_adam_variable_m_4)
%assignvariableop_20_adam_variable_m_5)
%assignvariableop_21_adam_variable_m_6'
#assignvariableop_22_adam_variable_v)
%assignvariableop_23_adam_variable_v_1)
%assignvariableop_24_adam_variable_v_2)
%assignvariableop_25_adam_variable_v_3)
%assignvariableop_26_adam_variable_v_4)
%assignvariableop_27_adam_variable_v_5)
%assignvariableop_28_adam_variable_v_6
identity_30ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBBWt_r/.ATTRIBUTES/VARIABLE_VALUEBWt_i/.ATTRIBUTES/VARIABLE_VALUEBWe_r/.ATTRIBUTES/VARIABLE_VALUEBWe_i/.ATTRIBUTES/VARIABLE_VALUEB alpha/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/0/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/1/.ATTRIBUTES/VARIABLE_VALUEB%lam_list/2/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB;Wt_r/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;Wt_i/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;We_r/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;We_i/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB;Wt_r/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB;Wt_i/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB;We_r/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB;We_i/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAlam_list/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЪ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesТ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ђ
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ђ
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ђ
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_3Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ђ
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_4Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ђ
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_5Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ђ
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_6Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ђ
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_7Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ё
AssignVariableOp_10AssignVariableOpassignvariableop_10_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Љ
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12Ѕ
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ё
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ё
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ћ
AssignVariableOp_15AssignVariableOp#assignvariableop_15_adam_variable_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16­
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_variable_m_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17­
AssignVariableOp_17AssignVariableOp%assignvariableop_17_adam_variable_m_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18­
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_variable_m_3Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19­
AssignVariableOp_19AssignVariableOp%assignvariableop_19_adam_variable_m_4Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20­
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_variable_m_5Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21­
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_variable_m_6Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ћ
AssignVariableOp_22AssignVariableOp#assignvariableop_22_adam_variable_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23­
AssignVariableOp_23AssignVariableOp%assignvariableop_23_adam_variable_v_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24­
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_variable_v_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25­
AssignVariableOp_25AssignVariableOp%assignvariableop_25_adam_variable_v_3Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26­
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_variable_v_4Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27­
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_variable_v_5Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28­
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_variable_v_6Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpм
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29Я
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*
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
_user_specified_namefile_prefix

Щ
(__inference_lista_3_layer_call_fn_903589
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identityЂStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџш*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lista_3_layer_call_and_return_conditional_losses_9035692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџ@:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
Іе

!__inference__wrapped_model_903013
input_1-
)lista_3_tensordot_readvariableop_resource+
'lista_3_maximum_readvariableop_resource/
+lista_3_tensordot_1_readvariableop_resource/
+lista_3_tensordot_2_readvariableop_resource/
+lista_3_tensordot_3_readvariableop_resource-
)lista_3_maximum_2_readvariableop_resource-
)lista_3_maximum_4_readvariableop_resource
identityZ
lista_3/RealRealinput_1*+
_output_shapes
:џџџџџџџџџ@2
lista_3/RealZ
lista_3/ImagImaginput_1*+
_output_shapes
:џџџџџџџџџ@2
lista_3/ImagЏ
 lista_3/Tensordot/ReadVariableOpReadVariableOp)lista_3_tensordot_readvariableop_resource*
_output_shapes
:	Д@*
dtype02"
 lista_3/Tensordot/ReadVariableOpz
lista_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot/axes
lista_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot/freew
lista_3/Tensordot/ShapeShapelista_3/Real:output:0*
T0*
_output_shapes
:2
lista_3/Tensordot/Shape
lista_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot/GatherV2/axisљ
lista_3/Tensordot/GatherV2GatherV2 lista_3/Tensordot/Shape:output:0lista_3/Tensordot/free:output:0(lista_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot/GatherV2
!lista_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot/GatherV2_1/axisџ
lista_3/Tensordot/GatherV2_1GatherV2 lista_3/Tensordot/Shape:output:0lista_3/Tensordot/axes:output:0*lista_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot/GatherV2_1|
lista_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot/Const 
lista_3/Tensordot/ProdProd#lista_3/Tensordot/GatherV2:output:0 lista_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot/Prod
lista_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot/Const_1Ј
lista_3/Tensordot/Prod_1Prod%lista_3/Tensordot/GatherV2_1:output:0"lista_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot/Prod_1
lista_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
lista_3/Tensordot/concat/axisи
lista_3/Tensordot/concatConcatV2lista_3/Tensordot/axes:output:0lista_3/Tensordot/free:output:0&lista_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot/concatЌ
lista_3/Tensordot/stackPack!lista_3/Tensordot/Prod_1:output:0lista_3/Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot/stackЗ
lista_3/Tensordot/transpose	Transposelista_3/Real:output:0!lista_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
lista_3/Tensordot/transposeП
lista_3/Tensordot/ReshapeReshapelista_3/Tensordot/transpose:y:0 lista_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot/ReshapeП
lista_3/Tensordot/MatMulMatMul(lista_3/Tensordot/ReadVariableOp:value:0"lista_3/Tensordot/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot/MatMul
lista_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot/Const_2
lista_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot/concat_1/axisх
lista_3/Tensordot/concat_1ConcatV2"lista_3/Tensordot/Const_2:output:0#lista_3/Tensordot/GatherV2:output:0(lista_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot/concat_1Б
lista_3/TensordotReshape"lista_3/Tensordot/MatMul:product:0#lista_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot
lista_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose/permЇ
lista_3/transpose	Transposelista_3/Tensordot:output:0lista_3/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose
lista_3/norm/mulMullista_3/transpose:y:0lista_3/transpose:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm/mul
"lista_3/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"lista_3/norm/Sum/reduction_indicesЖ
lista_3/norm/SumSumlista_3/norm/mul:z:0+lista_3/norm/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2
lista_3/norm/Sum
lista_3/norm/SqrtSqrtlista_3/norm/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm/Sqrt
lista_3/norm/SqueezeSqueezelista_3/norm/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
lista_3/norm/Squeezer
lista_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lista_3/ExpandDims/dim­
lista_3/ExpandDims
ExpandDimslista_3/norm/Squeeze:output:0lista_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/ExpandDims
lista_3/Tile/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
lista_3/Tile/multiples
lista_3/TileTilelista_3/ExpandDims:output:0lista_3/Tile/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Tile 
lista_3/Maximum/ReadVariableOpReadVariableOp'lista_3_maximum_readvariableop_resource*
_output_shapes
: *
dtype02 
lista_3/Maximum/ReadVariableOpЃ
lista_3/MaximumMaximumlista_3/Tile:output:0&lista_3/Maximum/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Maximum
lista_3/ReadVariableOpReadVariableOp'lista_3_maximum_readvariableop_resource*
_output_shapes
: *
dtype02
lista_3/ReadVariableOp
lista_3/truedivRealDivlista_3/ReadVariableOp:value:0lista_3/Maximum:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/truedivc
lista_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lista_3/sub/x
lista_3/subSublista_3/sub/x:output:0lista_3/truediv:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/sub
lista_3/mulMullista_3/sub:z:0lista_3/transpose:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/mulЕ
"lista_3/Tensordot_1/ReadVariableOpReadVariableOp+lista_3_tensordot_1_readvariableop_resource*
_output_shapes
:	Д@*
dtype02$
"lista_3/Tensordot_1/ReadVariableOp~
lista_3/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_1/axes
lista_3/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_1/free{
lista_3/Tensordot_1/ShapeShapelista_3/Imag:output:0*
T0*
_output_shapes
:2
lista_3/Tensordot_1/Shape
!lista_3/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_1/GatherV2/axis
lista_3/Tensordot_1/GatherV2GatherV2"lista_3/Tensordot_1/Shape:output:0!lista_3/Tensordot_1/free:output:0*lista_3/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_1/GatherV2
#lista_3/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#lista_3/Tensordot_1/GatherV2_1/axis
lista_3/Tensordot_1/GatherV2_1GatherV2"lista_3/Tensordot_1/Shape:output:0!lista_3/Tensordot_1/axes:output:0,lista_3/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
lista_3/Tensordot_1/GatherV2_1
lista_3/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_1/ConstЈ
lista_3/Tensordot_1/ProdProd%lista_3/Tensordot_1/GatherV2:output:0"lista_3/Tensordot_1/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_1/Prod
lista_3/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_1/Const_1А
lista_3/Tensordot_1/Prod_1Prod'lista_3/Tensordot_1/GatherV2_1:output:0$lista_3/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_1/Prod_1
lista_3/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot_1/concat/axisт
lista_3/Tensordot_1/concatConcatV2!lista_3/Tensordot_1/axes:output:0!lista_3/Tensordot_1/free:output:0(lista_3/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_1/concatД
lista_3/Tensordot_1/stackPack#lista_3/Tensordot_1/Prod_1:output:0!lista_3/Tensordot_1/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_1/stackН
lista_3/Tensordot_1/transpose	Transposelista_3/Imag:output:0#lista_3/Tensordot_1/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
lista_3/Tensordot_1/transposeЧ
lista_3/Tensordot_1/ReshapeReshape!lista_3/Tensordot_1/transpose:y:0"lista_3/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_1/ReshapeЧ
lista_3/Tensordot_1/MatMulMatMul*lista_3/Tensordot_1/ReadVariableOp:value:0$lista_3/Tensordot_1/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_1/MatMul
lista_3/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_1/Const_2
!lista_3/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_1/concat_1/axisя
lista_3/Tensordot_1/concat_1ConcatV2$lista_3/Tensordot_1/Const_2:output:0%lista_3/Tensordot_1/GatherV2:output:0*lista_3/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_1/concat_1Й
lista_3/Tensordot_1Reshape$lista_3/Tensordot_1/MatMul:product:0%lista_3/Tensordot_1/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_1
lista_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_1/permЏ
lista_3/transpose_1	Transposelista_3/Tensordot_1:output:0!lista_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_1
lista_3/norm_1/mulMullista_3/transpose_1:y:0lista_3/transpose_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm_1/mul
$lista_3/norm_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2&
$lista_3/norm_1/Sum/reduction_indicesО
lista_3/norm_1/SumSumlista_3/norm_1/mul:z:0-lista_3/norm_1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2
lista_3/norm_1/Sum
lista_3/norm_1/SqrtSqrtlista_3/norm_1/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm_1/Sqrt
lista_3/norm_1/SqueezeSqueezelista_3/norm_1/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
lista_3/norm_1/Squeezev
lista_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lista_3/ExpandDims_1/dimЕ
lista_3/ExpandDims_1
ExpandDimslista_3/norm_1/Squeeze:output:0!lista_3/ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/ExpandDims_1
lista_3/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
lista_3/Tile_1/multiplesЁ
lista_3/Tile_1Tilelista_3/ExpandDims_1:output:0!lista_3/Tile_1/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Tile_1Є
 lista_3/Maximum_1/ReadVariableOpReadVariableOp'lista_3_maximum_readvariableop_resource*
_output_shapes
: *
dtype02"
 lista_3/Maximum_1/ReadVariableOpЋ
lista_3/Maximum_1Maximumlista_3/Tile_1:output:0(lista_3/Maximum_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Maximum_1
lista_3/ReadVariableOp_1ReadVariableOp'lista_3_maximum_readvariableop_resource*
_output_shapes
: *
dtype02
lista_3/ReadVariableOp_1Ё
lista_3/truediv_1RealDiv lista_3/ReadVariableOp_1:value:0lista_3/Maximum_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/truediv_1g
lista_3/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lista_3/sub_1/x
lista_3/sub_1Sublista_3/sub_1/x:output:0lista_3/truediv_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/sub_1
lista_3/mul_1Mullista_3/sub_1:z:0lista_3/transpose_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/mul_1Ж
"lista_3/Tensordot_2/ReadVariableOpReadVariableOp+lista_3_tensordot_2_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02$
"lista_3/Tensordot_2/ReadVariableOp~
lista_3/Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_2/axes
lista_3/Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_2/freeu
lista_3/Tensordot_2/ShapeShapelista_3/mul:z:0*
T0*
_output_shapes
:2
lista_3/Tensordot_2/Shape
!lista_3/Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_2/GatherV2/axis
lista_3/Tensordot_2/GatherV2GatherV2"lista_3/Tensordot_2/Shape:output:0!lista_3/Tensordot_2/free:output:0*lista_3/Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_2/GatherV2
#lista_3/Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#lista_3/Tensordot_2/GatherV2_1/axis
lista_3/Tensordot_2/GatherV2_1GatherV2"lista_3/Tensordot_2/Shape:output:0!lista_3/Tensordot_2/axes:output:0,lista_3/Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
lista_3/Tensordot_2/GatherV2_1
lista_3/Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_2/ConstЈ
lista_3/Tensordot_2/ProdProd%lista_3/Tensordot_2/GatherV2:output:0"lista_3/Tensordot_2/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_2/Prod
lista_3/Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_2/Const_1А
lista_3/Tensordot_2/Prod_1Prod'lista_3/Tensordot_2/GatherV2_1:output:0$lista_3/Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_2/Prod_1
lista_3/Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot_2/concat/axisт
lista_3/Tensordot_2/concatConcatV2!lista_3/Tensordot_2/axes:output:0!lista_3/Tensordot_2/free:output:0(lista_3/Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_2/concatД
lista_3/Tensordot_2/stackPack#lista_3/Tensordot_2/Prod_1:output:0!lista_3/Tensordot_2/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_2/stackИ
lista_3/Tensordot_2/transpose	Transposelista_3/mul:z:0#lista_3/Tensordot_2/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_2/transposeЧ
lista_3/Tensordot_2/ReshapeReshape!lista_3/Tensordot_2/transpose:y:0"lista_3/Tensordot_2/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_2/ReshapeЧ
lista_3/Tensordot_2/MatMulMatMul*lista_3/Tensordot_2/ReadVariableOp:value:0$lista_3/Tensordot_2/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_2/MatMul
lista_3/Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_2/Const_2
!lista_3/Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_2/concat_1/axisя
lista_3/Tensordot_2/concat_1ConcatV2$lista_3/Tensordot_2/Const_2:output:0%lista_3/Tensordot_2/GatherV2:output:0*lista_3/Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_2/concat_1Й
lista_3/Tensordot_2Reshape$lista_3/Tensordot_2/MatMul:product:0%lista_3/Tensordot_2/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_2
lista_3/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_2/permЏ
lista_3/transpose_2	Transposelista_3/Tensordot_2:output:0!lista_3/transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_2Ж
"lista_3/Tensordot_3/ReadVariableOpReadVariableOp+lista_3_tensordot_3_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02$
"lista_3/Tensordot_3/ReadVariableOp~
lista_3/Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_3/axes
lista_3/Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_3/freew
lista_3/Tensordot_3/ShapeShapelista_3/mul_1:z:0*
T0*
_output_shapes
:2
lista_3/Tensordot_3/Shape
!lista_3/Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_3/GatherV2/axis
lista_3/Tensordot_3/GatherV2GatherV2"lista_3/Tensordot_3/Shape:output:0!lista_3/Tensordot_3/free:output:0*lista_3/Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_3/GatherV2
#lista_3/Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#lista_3/Tensordot_3/GatherV2_1/axis
lista_3/Tensordot_3/GatherV2_1GatherV2"lista_3/Tensordot_3/Shape:output:0!lista_3/Tensordot_3/axes:output:0,lista_3/Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
lista_3/Tensordot_3/GatherV2_1
lista_3/Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_3/ConstЈ
lista_3/Tensordot_3/ProdProd%lista_3/Tensordot_3/GatherV2:output:0"lista_3/Tensordot_3/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_3/Prod
lista_3/Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_3/Const_1А
lista_3/Tensordot_3/Prod_1Prod'lista_3/Tensordot_3/GatherV2_1:output:0$lista_3/Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_3/Prod_1
lista_3/Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot_3/concat/axisт
lista_3/Tensordot_3/concatConcatV2!lista_3/Tensordot_3/axes:output:0!lista_3/Tensordot_3/free:output:0(lista_3/Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_3/concatД
lista_3/Tensordot_3/stackPack#lista_3/Tensordot_3/Prod_1:output:0!lista_3/Tensordot_3/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_3/stackК
lista_3/Tensordot_3/transpose	Transposelista_3/mul_1:z:0#lista_3/Tensordot_3/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_3/transposeЧ
lista_3/Tensordot_3/ReshapeReshape!lista_3/Tensordot_3/transpose:y:0"lista_3/Tensordot_3/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_3/ReshapeЧ
lista_3/Tensordot_3/MatMulMatMul*lista_3/Tensordot_3/ReadVariableOp:value:0$lista_3/Tensordot_3/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_3/MatMul
lista_3/Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_3/Const_2
!lista_3/Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_3/concat_1/axisя
lista_3/Tensordot_3/concat_1ConcatV2$lista_3/Tensordot_3/Const_2:output:0%lista_3/Tensordot_3/GatherV2:output:0*lista_3/Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_3/concat_1Й
lista_3/Tensordot_3Reshape$lista_3/Tensordot_3/MatMul:product:0%lista_3/Tensordot_3/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_3
lista_3/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_3/permЏ
lista_3/transpose_3	Transposelista_3/Tensordot_3:output:0!lista_3/transpose_3/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_3
lista_3/sub_2Sublista_3/transpose_2:y:0lista_3/transpose_3:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/sub_2Г
"lista_3/Tensordot_4/ReadVariableOpReadVariableOp)lista_3_tensordot_readvariableop_resource*
_output_shapes
:	Д@*
dtype02$
"lista_3/Tensordot_4/ReadVariableOp~
lista_3/Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_4/axes
lista_3/Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_4/free{
lista_3/Tensordot_4/ShapeShapelista_3/Real:output:0*
T0*
_output_shapes
:2
lista_3/Tensordot_4/Shape
!lista_3/Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_4/GatherV2/axis
lista_3/Tensordot_4/GatherV2GatherV2"lista_3/Tensordot_4/Shape:output:0!lista_3/Tensordot_4/free:output:0*lista_3/Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_4/GatherV2
#lista_3/Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#lista_3/Tensordot_4/GatherV2_1/axis
lista_3/Tensordot_4/GatherV2_1GatherV2"lista_3/Tensordot_4/Shape:output:0!lista_3/Tensordot_4/axes:output:0,lista_3/Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
lista_3/Tensordot_4/GatherV2_1
lista_3/Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_4/ConstЈ
lista_3/Tensordot_4/ProdProd%lista_3/Tensordot_4/GatherV2:output:0"lista_3/Tensordot_4/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_4/Prod
lista_3/Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_4/Const_1А
lista_3/Tensordot_4/Prod_1Prod'lista_3/Tensordot_4/GatherV2_1:output:0$lista_3/Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_4/Prod_1
lista_3/Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot_4/concat/axisт
lista_3/Tensordot_4/concatConcatV2!lista_3/Tensordot_4/axes:output:0!lista_3/Tensordot_4/free:output:0(lista_3/Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_4/concatД
lista_3/Tensordot_4/stackPack#lista_3/Tensordot_4/Prod_1:output:0!lista_3/Tensordot_4/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_4/stackН
lista_3/Tensordot_4/transpose	Transposelista_3/Real:output:0#lista_3/Tensordot_4/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
lista_3/Tensordot_4/transposeЧ
lista_3/Tensordot_4/ReshapeReshape!lista_3/Tensordot_4/transpose:y:0"lista_3/Tensordot_4/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_4/ReshapeЧ
lista_3/Tensordot_4/MatMulMatMul*lista_3/Tensordot_4/ReadVariableOp:value:0$lista_3/Tensordot_4/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_4/MatMul
lista_3/Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_4/Const_2
!lista_3/Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_4/concat_1/axisя
lista_3/Tensordot_4/concat_1ConcatV2$lista_3/Tensordot_4/Const_2:output:0%lista_3/Tensordot_4/GatherV2:output:0*lista_3/Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_4/concat_1Й
lista_3/Tensordot_4Reshape$lista_3/Tensordot_4/MatMul:product:0%lista_3/Tensordot_4/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_4
lista_3/transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_4/permЏ
lista_3/transpose_4	Transposelista_3/Tensordot_4:output:0!lista_3/transpose_4/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_4
lista_3/addAddV2lista_3/sub_2:z:0lista_3/transpose_4:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/addЕ
"lista_3/Tensordot_5/ReadVariableOpReadVariableOp+lista_3_tensordot_1_readvariableop_resource*
_output_shapes
:	Д@*
dtype02$
"lista_3/Tensordot_5/ReadVariableOp~
lista_3/Tensordot_5/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_5/axes
lista_3/Tensordot_5/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_5/free{
lista_3/Tensordot_5/ShapeShapelista_3/Imag:output:0*
T0*
_output_shapes
:2
lista_3/Tensordot_5/Shape
!lista_3/Tensordot_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_5/GatherV2/axis
lista_3/Tensordot_5/GatherV2GatherV2"lista_3/Tensordot_5/Shape:output:0!lista_3/Tensordot_5/free:output:0*lista_3/Tensordot_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_5/GatherV2
#lista_3/Tensordot_5/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#lista_3/Tensordot_5/GatherV2_1/axis
lista_3/Tensordot_5/GatherV2_1GatherV2"lista_3/Tensordot_5/Shape:output:0!lista_3/Tensordot_5/axes:output:0,lista_3/Tensordot_5/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
lista_3/Tensordot_5/GatherV2_1
lista_3/Tensordot_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_5/ConstЈ
lista_3/Tensordot_5/ProdProd%lista_3/Tensordot_5/GatherV2:output:0"lista_3/Tensordot_5/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_5/Prod
lista_3/Tensordot_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_5/Const_1А
lista_3/Tensordot_5/Prod_1Prod'lista_3/Tensordot_5/GatherV2_1:output:0$lista_3/Tensordot_5/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_5/Prod_1
lista_3/Tensordot_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot_5/concat/axisт
lista_3/Tensordot_5/concatConcatV2!lista_3/Tensordot_5/axes:output:0!lista_3/Tensordot_5/free:output:0(lista_3/Tensordot_5/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_5/concatД
lista_3/Tensordot_5/stackPack#lista_3/Tensordot_5/Prod_1:output:0!lista_3/Tensordot_5/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_5/stackН
lista_3/Tensordot_5/transpose	Transposelista_3/Imag:output:0#lista_3/Tensordot_5/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
lista_3/Tensordot_5/transposeЧ
lista_3/Tensordot_5/ReshapeReshape!lista_3/Tensordot_5/transpose:y:0"lista_3/Tensordot_5/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_5/ReshapeЧ
lista_3/Tensordot_5/MatMulMatMul*lista_3/Tensordot_5/ReadVariableOp:value:0$lista_3/Tensordot_5/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_5/MatMul
lista_3/Tensordot_5/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_5/Const_2
!lista_3/Tensordot_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_5/concat_1/axisя
lista_3/Tensordot_5/concat_1ConcatV2$lista_3/Tensordot_5/Const_2:output:0%lista_3/Tensordot_5/GatherV2:output:0*lista_3/Tensordot_5/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_5/concat_1Й
lista_3/Tensordot_5Reshape$lista_3/Tensordot_5/MatMul:product:0%lista_3/Tensordot_5/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_5
lista_3/transpose_5/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_5/permЏ
lista_3/transpose_5	Transposelista_3/Tensordot_5:output:0!lista_3/transpose_5/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_5
lista_3/sub_3Sublista_3/add:z:0lista_3/transpose_5:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/sub_3
lista_3/norm_2/mulMullista_3/sub_3:z:0lista_3/sub_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm_2/mul
$lista_3/norm_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2&
$lista_3/norm_2/Sum/reduction_indicesО
lista_3/norm_2/SumSumlista_3/norm_2/mul:z:0-lista_3/norm_2/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2
lista_3/norm_2/Sum
lista_3/norm_2/SqrtSqrtlista_3/norm_2/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm_2/Sqrt
lista_3/norm_2/SqueezeSqueezelista_3/norm_2/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
lista_3/norm_2/Squeezev
lista_3/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lista_3/ExpandDims_2/dimЕ
lista_3/ExpandDims_2
ExpandDimslista_3/norm_2/Squeeze:output:0!lista_3/ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/ExpandDims_2
lista_3/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
lista_3/Tile_2/multiplesЁ
lista_3/Tile_2Tilelista_3/ExpandDims_2:output:0!lista_3/Tile_2/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Tile_2І
 lista_3/Maximum_2/ReadVariableOpReadVariableOp)lista_3_maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02"
 lista_3/Maximum_2/ReadVariableOpЋ
lista_3/Maximum_2Maximumlista_3/Tile_2:output:0(lista_3/Maximum_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Maximum_2
lista_3/ReadVariableOp_2ReadVariableOp)lista_3_maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02
lista_3/ReadVariableOp_2Ё
lista_3/truediv_2RealDiv lista_3/ReadVariableOp_2:value:0lista_3/Maximum_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/truediv_2g
lista_3/sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lista_3/sub_4/x
lista_3/sub_4Sublista_3/sub_4/x:output:0lista_3/truediv_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/sub_4
lista_3/mul_2Mullista_3/sub_4:z:0lista_3/sub_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/mul_2Ж
"lista_3/Tensordot_6/ReadVariableOpReadVariableOp+lista_3_tensordot_2_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02$
"lista_3/Tensordot_6/ReadVariableOp~
lista_3/Tensordot_6/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_6/axes
lista_3/Tensordot_6/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_6/freew
lista_3/Tensordot_6/ShapeShapelista_3/mul_1:z:0*
T0*
_output_shapes
:2
lista_3/Tensordot_6/Shape
!lista_3/Tensordot_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_6/GatherV2/axis
lista_3/Tensordot_6/GatherV2GatherV2"lista_3/Tensordot_6/Shape:output:0!lista_3/Tensordot_6/free:output:0*lista_3/Tensordot_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_6/GatherV2
#lista_3/Tensordot_6/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#lista_3/Tensordot_6/GatherV2_1/axis
lista_3/Tensordot_6/GatherV2_1GatherV2"lista_3/Tensordot_6/Shape:output:0!lista_3/Tensordot_6/axes:output:0,lista_3/Tensordot_6/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
lista_3/Tensordot_6/GatherV2_1
lista_3/Tensordot_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_6/ConstЈ
lista_3/Tensordot_6/ProdProd%lista_3/Tensordot_6/GatherV2:output:0"lista_3/Tensordot_6/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_6/Prod
lista_3/Tensordot_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_6/Const_1А
lista_3/Tensordot_6/Prod_1Prod'lista_3/Tensordot_6/GatherV2_1:output:0$lista_3/Tensordot_6/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_6/Prod_1
lista_3/Tensordot_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot_6/concat/axisт
lista_3/Tensordot_6/concatConcatV2!lista_3/Tensordot_6/axes:output:0!lista_3/Tensordot_6/free:output:0(lista_3/Tensordot_6/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_6/concatД
lista_3/Tensordot_6/stackPack#lista_3/Tensordot_6/Prod_1:output:0!lista_3/Tensordot_6/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_6/stackК
lista_3/Tensordot_6/transpose	Transposelista_3/mul_1:z:0#lista_3/Tensordot_6/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_6/transposeЧ
lista_3/Tensordot_6/ReshapeReshape!lista_3/Tensordot_6/transpose:y:0"lista_3/Tensordot_6/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_6/ReshapeЧ
lista_3/Tensordot_6/MatMulMatMul*lista_3/Tensordot_6/ReadVariableOp:value:0$lista_3/Tensordot_6/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_6/MatMul
lista_3/Tensordot_6/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_6/Const_2
!lista_3/Tensordot_6/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_6/concat_1/axisя
lista_3/Tensordot_6/concat_1ConcatV2$lista_3/Tensordot_6/Const_2:output:0%lista_3/Tensordot_6/GatherV2:output:0*lista_3/Tensordot_6/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_6/concat_1Й
lista_3/Tensordot_6Reshape$lista_3/Tensordot_6/MatMul:product:0%lista_3/Tensordot_6/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_6
lista_3/transpose_6/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_6/permЏ
lista_3/transpose_6	Transposelista_3/Tensordot_6:output:0!lista_3/transpose_6/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_6Ж
"lista_3/Tensordot_7/ReadVariableOpReadVariableOp+lista_3_tensordot_3_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02$
"lista_3/Tensordot_7/ReadVariableOp~
lista_3/Tensordot_7/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_7/axes
lista_3/Tensordot_7/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_7/freew
lista_3/Tensordot_7/ShapeShapelista_3/mul_2:z:0*
T0*
_output_shapes
:2
lista_3/Tensordot_7/Shape
!lista_3/Tensordot_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_7/GatherV2/axis
lista_3/Tensordot_7/GatherV2GatherV2"lista_3/Tensordot_7/Shape:output:0!lista_3/Tensordot_7/free:output:0*lista_3/Tensordot_7/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_7/GatherV2
#lista_3/Tensordot_7/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#lista_3/Tensordot_7/GatherV2_1/axis
lista_3/Tensordot_7/GatherV2_1GatherV2"lista_3/Tensordot_7/Shape:output:0!lista_3/Tensordot_7/axes:output:0,lista_3/Tensordot_7/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
lista_3/Tensordot_7/GatherV2_1
lista_3/Tensordot_7/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_7/ConstЈ
lista_3/Tensordot_7/ProdProd%lista_3/Tensordot_7/GatherV2:output:0"lista_3/Tensordot_7/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_7/Prod
lista_3/Tensordot_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_7/Const_1А
lista_3/Tensordot_7/Prod_1Prod'lista_3/Tensordot_7/GatherV2_1:output:0$lista_3/Tensordot_7/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_7/Prod_1
lista_3/Tensordot_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot_7/concat/axisт
lista_3/Tensordot_7/concatConcatV2!lista_3/Tensordot_7/axes:output:0!lista_3/Tensordot_7/free:output:0(lista_3/Tensordot_7/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_7/concatД
lista_3/Tensordot_7/stackPack#lista_3/Tensordot_7/Prod_1:output:0!lista_3/Tensordot_7/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_7/stackК
lista_3/Tensordot_7/transpose	Transposelista_3/mul_2:z:0#lista_3/Tensordot_7/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_7/transposeЧ
lista_3/Tensordot_7/ReshapeReshape!lista_3/Tensordot_7/transpose:y:0"lista_3/Tensordot_7/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_7/ReshapeЧ
lista_3/Tensordot_7/MatMulMatMul*lista_3/Tensordot_7/ReadVariableOp:value:0$lista_3/Tensordot_7/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_7/MatMul
lista_3/Tensordot_7/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_7/Const_2
!lista_3/Tensordot_7/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_7/concat_1/axisя
lista_3/Tensordot_7/concat_1ConcatV2$lista_3/Tensordot_7/Const_2:output:0%lista_3/Tensordot_7/GatherV2:output:0*lista_3/Tensordot_7/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_7/concat_1Й
lista_3/Tensordot_7Reshape$lista_3/Tensordot_7/MatMul:product:0%lista_3/Tensordot_7/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_7
lista_3/transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_7/permЏ
lista_3/transpose_7	Transposelista_3/Tensordot_7:output:0!lista_3/transpose_7/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_7
lista_3/add_1AddV2lista_3/transpose_6:y:0lista_3/transpose_7:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/add_1Г
"lista_3/Tensordot_8/ReadVariableOpReadVariableOp)lista_3_tensordot_readvariableop_resource*
_output_shapes
:	Д@*
dtype02$
"lista_3/Tensordot_8/ReadVariableOp~
lista_3/Tensordot_8/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_8/axes
lista_3/Tensordot_8/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_8/free{
lista_3/Tensordot_8/ShapeShapelista_3/Imag:output:0*
T0*
_output_shapes
:2
lista_3/Tensordot_8/Shape
!lista_3/Tensordot_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_8/GatherV2/axis
lista_3/Tensordot_8/GatherV2GatherV2"lista_3/Tensordot_8/Shape:output:0!lista_3/Tensordot_8/free:output:0*lista_3/Tensordot_8/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_8/GatherV2
#lista_3/Tensordot_8/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#lista_3/Tensordot_8/GatherV2_1/axis
lista_3/Tensordot_8/GatherV2_1GatherV2"lista_3/Tensordot_8/Shape:output:0!lista_3/Tensordot_8/axes:output:0,lista_3/Tensordot_8/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
lista_3/Tensordot_8/GatherV2_1
lista_3/Tensordot_8/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_8/ConstЈ
lista_3/Tensordot_8/ProdProd%lista_3/Tensordot_8/GatherV2:output:0"lista_3/Tensordot_8/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_8/Prod
lista_3/Tensordot_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_8/Const_1А
lista_3/Tensordot_8/Prod_1Prod'lista_3/Tensordot_8/GatherV2_1:output:0$lista_3/Tensordot_8/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_8/Prod_1
lista_3/Tensordot_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot_8/concat/axisт
lista_3/Tensordot_8/concatConcatV2!lista_3/Tensordot_8/axes:output:0!lista_3/Tensordot_8/free:output:0(lista_3/Tensordot_8/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_8/concatД
lista_3/Tensordot_8/stackPack#lista_3/Tensordot_8/Prod_1:output:0!lista_3/Tensordot_8/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_8/stackН
lista_3/Tensordot_8/transpose	Transposelista_3/Imag:output:0#lista_3/Tensordot_8/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
lista_3/Tensordot_8/transposeЧ
lista_3/Tensordot_8/ReshapeReshape!lista_3/Tensordot_8/transpose:y:0"lista_3/Tensordot_8/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_8/ReshapeЧ
lista_3/Tensordot_8/MatMulMatMul*lista_3/Tensordot_8/ReadVariableOp:value:0$lista_3/Tensordot_8/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_8/MatMul
lista_3/Tensordot_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_8/Const_2
!lista_3/Tensordot_8/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_8/concat_1/axisя
lista_3/Tensordot_8/concat_1ConcatV2$lista_3/Tensordot_8/Const_2:output:0%lista_3/Tensordot_8/GatherV2:output:0*lista_3/Tensordot_8/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_8/concat_1Й
lista_3/Tensordot_8Reshape$lista_3/Tensordot_8/MatMul:product:0%lista_3/Tensordot_8/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_8
lista_3/transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_8/permЏ
lista_3/transpose_8	Transposelista_3/Tensordot_8:output:0!lista_3/transpose_8/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_8
lista_3/add_2AddV2lista_3/add_1:z:0lista_3/transpose_8:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/add_2Е
"lista_3/Tensordot_9/ReadVariableOpReadVariableOp+lista_3_tensordot_1_readvariableop_resource*
_output_shapes
:	Д@*
dtype02$
"lista_3/Tensordot_9/ReadVariableOp~
lista_3/Tensordot_9/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_9/axes
lista_3/Tensordot_9/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_9/free{
lista_3/Tensordot_9/ShapeShapelista_3/Real:output:0*
T0*
_output_shapes
:2
lista_3/Tensordot_9/Shape
!lista_3/Tensordot_9/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_9/GatherV2/axis
lista_3/Tensordot_9/GatherV2GatherV2"lista_3/Tensordot_9/Shape:output:0!lista_3/Tensordot_9/free:output:0*lista_3/Tensordot_9/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_9/GatherV2
#lista_3/Tensordot_9/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#lista_3/Tensordot_9/GatherV2_1/axis
lista_3/Tensordot_9/GatherV2_1GatherV2"lista_3/Tensordot_9/Shape:output:0!lista_3/Tensordot_9/axes:output:0,lista_3/Tensordot_9/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
lista_3/Tensordot_9/GatherV2_1
lista_3/Tensordot_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_9/ConstЈ
lista_3/Tensordot_9/ProdProd%lista_3/Tensordot_9/GatherV2:output:0"lista_3/Tensordot_9/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_9/Prod
lista_3/Tensordot_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_9/Const_1А
lista_3/Tensordot_9/Prod_1Prod'lista_3/Tensordot_9/GatherV2_1:output:0$lista_3/Tensordot_9/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_9/Prod_1
lista_3/Tensordot_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
lista_3/Tensordot_9/concat/axisт
lista_3/Tensordot_9/concatConcatV2!lista_3/Tensordot_9/axes:output:0!lista_3/Tensordot_9/free:output:0(lista_3/Tensordot_9/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_9/concatД
lista_3/Tensordot_9/stackPack#lista_3/Tensordot_9/Prod_1:output:0!lista_3/Tensordot_9/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_9/stackН
lista_3/Tensordot_9/transpose	Transposelista_3/Real:output:0#lista_3/Tensordot_9/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
lista_3/Tensordot_9/transposeЧ
lista_3/Tensordot_9/ReshapeReshape!lista_3/Tensordot_9/transpose:y:0"lista_3/Tensordot_9/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_9/ReshapeЧ
lista_3/Tensordot_9/MatMulMatMul*lista_3/Tensordot_9/ReadVariableOp:value:0$lista_3/Tensordot_9/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_9/MatMul
lista_3/Tensordot_9/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_9/Const_2
!lista_3/Tensordot_9/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!lista_3/Tensordot_9/concat_1/axisя
lista_3/Tensordot_9/concat_1ConcatV2$lista_3/Tensordot_9/Const_2:output:0%lista_3/Tensordot_9/GatherV2:output:0*lista_3/Tensordot_9/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_9/concat_1Й
lista_3/Tensordot_9Reshape$lista_3/Tensordot_9/MatMul:product:0%lista_3/Tensordot_9/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_9
lista_3/transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_9/permЏ
lista_3/transpose_9	Transposelista_3/Tensordot_9:output:0!lista_3/transpose_9/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_9
lista_3/add_3AddV2lista_3/add_2:z:0lista_3/transpose_9:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/add_3
lista_3/norm_3/mulMullista_3/add_3:z:0lista_3/add_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm_3/mul
$lista_3/norm_3/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2&
$lista_3/norm_3/Sum/reduction_indicesО
lista_3/norm_3/SumSumlista_3/norm_3/mul:z:0-lista_3/norm_3/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2
lista_3/norm_3/Sum
lista_3/norm_3/SqrtSqrtlista_3/norm_3/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm_3/Sqrt
lista_3/norm_3/SqueezeSqueezelista_3/norm_3/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
lista_3/norm_3/Squeezev
lista_3/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lista_3/ExpandDims_3/dimЕ
lista_3/ExpandDims_3
ExpandDimslista_3/norm_3/Squeeze:output:0!lista_3/ExpandDims_3/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/ExpandDims_3
lista_3/Tile_3/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
lista_3/Tile_3/multiplesЁ
lista_3/Tile_3Tilelista_3/ExpandDims_3:output:0!lista_3/Tile_3/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Tile_3І
 lista_3/Maximum_3/ReadVariableOpReadVariableOp)lista_3_maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02"
 lista_3/Maximum_3/ReadVariableOpЋ
lista_3/Maximum_3Maximumlista_3/Tile_3:output:0(lista_3/Maximum_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Maximum_3
lista_3/ReadVariableOp_3ReadVariableOp)lista_3_maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02
lista_3/ReadVariableOp_3Ё
lista_3/truediv_3RealDiv lista_3/ReadVariableOp_3:value:0lista_3/Maximum_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/truediv_3g
lista_3/sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lista_3/sub_5/x
lista_3/sub_5Sublista_3/sub_5/x:output:0lista_3/truediv_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/sub_5
lista_3/mul_3Mullista_3/sub_5:z:0lista_3/add_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/mul_3И
#lista_3/Tensordot_10/ReadVariableOpReadVariableOp+lista_3_tensordot_2_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02%
#lista_3/Tensordot_10/ReadVariableOp
lista_3/Tensordot_10/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_10/axes
lista_3/Tensordot_10/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_10/freey
lista_3/Tensordot_10/ShapeShapelista_3/mul_2:z:0*
T0*
_output_shapes
:2
lista_3/Tensordot_10/Shape
"lista_3/Tensordot_10/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_10/GatherV2/axis
lista_3/Tensordot_10/GatherV2GatherV2#lista_3/Tensordot_10/Shape:output:0"lista_3/Tensordot_10/free:output:0+lista_3/Tensordot_10/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_10/GatherV2
$lista_3/Tensordot_10/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lista_3/Tensordot_10/GatherV2_1/axis
lista_3/Tensordot_10/GatherV2_1GatherV2#lista_3/Tensordot_10/Shape:output:0"lista_3/Tensordot_10/axes:output:0-lista_3/Tensordot_10/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
lista_3/Tensordot_10/GatherV2_1
lista_3/Tensordot_10/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_10/ConstЌ
lista_3/Tensordot_10/ProdProd&lista_3/Tensordot_10/GatherV2:output:0#lista_3/Tensordot_10/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_10/Prod
lista_3/Tensordot_10/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_10/Const_1Д
lista_3/Tensordot_10/Prod_1Prod(lista_3/Tensordot_10/GatherV2_1:output:0%lista_3/Tensordot_10/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_10/Prod_1
 lista_3/Tensordot_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lista_3/Tensordot_10/concat/axisч
lista_3/Tensordot_10/concatConcatV2"lista_3/Tensordot_10/axes:output:0"lista_3/Tensordot_10/free:output:0)lista_3/Tensordot_10/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_10/concatИ
lista_3/Tensordot_10/stackPack$lista_3/Tensordot_10/Prod_1:output:0"lista_3/Tensordot_10/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_10/stackН
lista_3/Tensordot_10/transpose	Transposelista_3/mul_2:z:0$lista_3/Tensordot_10/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2 
lista_3/Tensordot_10/transposeЫ
lista_3/Tensordot_10/ReshapeReshape"lista_3/Tensordot_10/transpose:y:0#lista_3/Tensordot_10/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_10/ReshapeЫ
lista_3/Tensordot_10/MatMulMatMul+lista_3/Tensordot_10/ReadVariableOp:value:0%lista_3/Tensordot_10/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_10/MatMul
lista_3/Tensordot_10/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_10/Const_2
"lista_3/Tensordot_10/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_10/concat_1/axisє
lista_3/Tensordot_10/concat_1ConcatV2%lista_3/Tensordot_10/Const_2:output:0&lista_3/Tensordot_10/GatherV2:output:0+lista_3/Tensordot_10/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_10/concat_1Н
lista_3/Tensordot_10Reshape%lista_3/Tensordot_10/MatMul:product:0&lista_3/Tensordot_10/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_10
lista_3/transpose_10/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_10/permГ
lista_3/transpose_10	Transposelista_3/Tensordot_10:output:0"lista_3/transpose_10/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_10И
#lista_3/Tensordot_11/ReadVariableOpReadVariableOp+lista_3_tensordot_3_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02%
#lista_3/Tensordot_11/ReadVariableOp
lista_3/Tensordot_11/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_11/axes
lista_3/Tensordot_11/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_11/freey
lista_3/Tensordot_11/ShapeShapelista_3/mul_3:z:0*
T0*
_output_shapes
:2
lista_3/Tensordot_11/Shape
"lista_3/Tensordot_11/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_11/GatherV2/axis
lista_3/Tensordot_11/GatherV2GatherV2#lista_3/Tensordot_11/Shape:output:0"lista_3/Tensordot_11/free:output:0+lista_3/Tensordot_11/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_11/GatherV2
$lista_3/Tensordot_11/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lista_3/Tensordot_11/GatherV2_1/axis
lista_3/Tensordot_11/GatherV2_1GatherV2#lista_3/Tensordot_11/Shape:output:0"lista_3/Tensordot_11/axes:output:0-lista_3/Tensordot_11/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
lista_3/Tensordot_11/GatherV2_1
lista_3/Tensordot_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_11/ConstЌ
lista_3/Tensordot_11/ProdProd&lista_3/Tensordot_11/GatherV2:output:0#lista_3/Tensordot_11/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_11/Prod
lista_3/Tensordot_11/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_11/Const_1Д
lista_3/Tensordot_11/Prod_1Prod(lista_3/Tensordot_11/GatherV2_1:output:0%lista_3/Tensordot_11/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_11/Prod_1
 lista_3/Tensordot_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lista_3/Tensordot_11/concat/axisч
lista_3/Tensordot_11/concatConcatV2"lista_3/Tensordot_11/axes:output:0"lista_3/Tensordot_11/free:output:0)lista_3/Tensordot_11/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_11/concatИ
lista_3/Tensordot_11/stackPack$lista_3/Tensordot_11/Prod_1:output:0"lista_3/Tensordot_11/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_11/stackН
lista_3/Tensordot_11/transpose	Transposelista_3/mul_3:z:0$lista_3/Tensordot_11/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2 
lista_3/Tensordot_11/transposeЫ
lista_3/Tensordot_11/ReshapeReshape"lista_3/Tensordot_11/transpose:y:0#lista_3/Tensordot_11/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_11/ReshapeЫ
lista_3/Tensordot_11/MatMulMatMul+lista_3/Tensordot_11/ReadVariableOp:value:0%lista_3/Tensordot_11/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_11/MatMul
lista_3/Tensordot_11/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_11/Const_2
"lista_3/Tensordot_11/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_11/concat_1/axisє
lista_3/Tensordot_11/concat_1ConcatV2%lista_3/Tensordot_11/Const_2:output:0&lista_3/Tensordot_11/GatherV2:output:0+lista_3/Tensordot_11/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_11/concat_1Н
lista_3/Tensordot_11Reshape%lista_3/Tensordot_11/MatMul:product:0&lista_3/Tensordot_11/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_11
lista_3/transpose_11/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_11/permГ
lista_3/transpose_11	Transposelista_3/Tensordot_11:output:0"lista_3/transpose_11/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_11
lista_3/sub_6Sublista_3/transpose_10:y:0lista_3/transpose_11:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/sub_6Е
#lista_3/Tensordot_12/ReadVariableOpReadVariableOp)lista_3_tensordot_readvariableop_resource*
_output_shapes
:	Д@*
dtype02%
#lista_3/Tensordot_12/ReadVariableOp
lista_3/Tensordot_12/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_12/axes
lista_3/Tensordot_12/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_12/free}
lista_3/Tensordot_12/ShapeShapelista_3/Real:output:0*
T0*
_output_shapes
:2
lista_3/Tensordot_12/Shape
"lista_3/Tensordot_12/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_12/GatherV2/axis
lista_3/Tensordot_12/GatherV2GatherV2#lista_3/Tensordot_12/Shape:output:0"lista_3/Tensordot_12/free:output:0+lista_3/Tensordot_12/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_12/GatherV2
$lista_3/Tensordot_12/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lista_3/Tensordot_12/GatherV2_1/axis
lista_3/Tensordot_12/GatherV2_1GatherV2#lista_3/Tensordot_12/Shape:output:0"lista_3/Tensordot_12/axes:output:0-lista_3/Tensordot_12/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
lista_3/Tensordot_12/GatherV2_1
lista_3/Tensordot_12/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_12/ConstЌ
lista_3/Tensordot_12/ProdProd&lista_3/Tensordot_12/GatherV2:output:0#lista_3/Tensordot_12/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_12/Prod
lista_3/Tensordot_12/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_12/Const_1Д
lista_3/Tensordot_12/Prod_1Prod(lista_3/Tensordot_12/GatherV2_1:output:0%lista_3/Tensordot_12/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_12/Prod_1
 lista_3/Tensordot_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lista_3/Tensordot_12/concat/axisч
lista_3/Tensordot_12/concatConcatV2"lista_3/Tensordot_12/axes:output:0"lista_3/Tensordot_12/free:output:0)lista_3/Tensordot_12/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_12/concatИ
lista_3/Tensordot_12/stackPack$lista_3/Tensordot_12/Prod_1:output:0"lista_3/Tensordot_12/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_12/stackР
lista_3/Tensordot_12/transpose	Transposelista_3/Real:output:0$lista_3/Tensordot_12/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2 
lista_3/Tensordot_12/transposeЫ
lista_3/Tensordot_12/ReshapeReshape"lista_3/Tensordot_12/transpose:y:0#lista_3/Tensordot_12/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_12/ReshapeЫ
lista_3/Tensordot_12/MatMulMatMul+lista_3/Tensordot_12/ReadVariableOp:value:0%lista_3/Tensordot_12/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_12/MatMul
lista_3/Tensordot_12/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_12/Const_2
"lista_3/Tensordot_12/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_12/concat_1/axisє
lista_3/Tensordot_12/concat_1ConcatV2%lista_3/Tensordot_12/Const_2:output:0&lista_3/Tensordot_12/GatherV2:output:0+lista_3/Tensordot_12/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_12/concat_1Н
lista_3/Tensordot_12Reshape%lista_3/Tensordot_12/MatMul:product:0&lista_3/Tensordot_12/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_12
lista_3/transpose_12/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_12/permГ
lista_3/transpose_12	Transposelista_3/Tensordot_12:output:0"lista_3/transpose_12/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_12
lista_3/add_4AddV2lista_3/sub_6:z:0lista_3/transpose_12:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/add_4З
#lista_3/Tensordot_13/ReadVariableOpReadVariableOp+lista_3_tensordot_1_readvariableop_resource*
_output_shapes
:	Д@*
dtype02%
#lista_3/Tensordot_13/ReadVariableOp
lista_3/Tensordot_13/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_13/axes
lista_3/Tensordot_13/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_13/free}
lista_3/Tensordot_13/ShapeShapelista_3/Imag:output:0*
T0*
_output_shapes
:2
lista_3/Tensordot_13/Shape
"lista_3/Tensordot_13/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_13/GatherV2/axis
lista_3/Tensordot_13/GatherV2GatherV2#lista_3/Tensordot_13/Shape:output:0"lista_3/Tensordot_13/free:output:0+lista_3/Tensordot_13/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_13/GatherV2
$lista_3/Tensordot_13/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lista_3/Tensordot_13/GatherV2_1/axis
lista_3/Tensordot_13/GatherV2_1GatherV2#lista_3/Tensordot_13/Shape:output:0"lista_3/Tensordot_13/axes:output:0-lista_3/Tensordot_13/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
lista_3/Tensordot_13/GatherV2_1
lista_3/Tensordot_13/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_13/ConstЌ
lista_3/Tensordot_13/ProdProd&lista_3/Tensordot_13/GatherV2:output:0#lista_3/Tensordot_13/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_13/Prod
lista_3/Tensordot_13/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_13/Const_1Д
lista_3/Tensordot_13/Prod_1Prod(lista_3/Tensordot_13/GatherV2_1:output:0%lista_3/Tensordot_13/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_13/Prod_1
 lista_3/Tensordot_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lista_3/Tensordot_13/concat/axisч
lista_3/Tensordot_13/concatConcatV2"lista_3/Tensordot_13/axes:output:0"lista_3/Tensordot_13/free:output:0)lista_3/Tensordot_13/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_13/concatИ
lista_3/Tensordot_13/stackPack$lista_3/Tensordot_13/Prod_1:output:0"lista_3/Tensordot_13/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_13/stackР
lista_3/Tensordot_13/transpose	Transposelista_3/Imag:output:0$lista_3/Tensordot_13/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2 
lista_3/Tensordot_13/transposeЫ
lista_3/Tensordot_13/ReshapeReshape"lista_3/Tensordot_13/transpose:y:0#lista_3/Tensordot_13/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_13/ReshapeЫ
lista_3/Tensordot_13/MatMulMatMul+lista_3/Tensordot_13/ReadVariableOp:value:0%lista_3/Tensordot_13/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_13/MatMul
lista_3/Tensordot_13/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_13/Const_2
"lista_3/Tensordot_13/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_13/concat_1/axisє
lista_3/Tensordot_13/concat_1ConcatV2%lista_3/Tensordot_13/Const_2:output:0&lista_3/Tensordot_13/GatherV2:output:0+lista_3/Tensordot_13/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_13/concat_1Н
lista_3/Tensordot_13Reshape%lista_3/Tensordot_13/MatMul:product:0&lista_3/Tensordot_13/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_13
lista_3/transpose_13/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_13/permГ
lista_3/transpose_13	Transposelista_3/Tensordot_13:output:0"lista_3/transpose_13/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_13
lista_3/sub_7Sublista_3/add_4:z:0lista_3/transpose_13:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/sub_7
lista_3/norm_4/mulMullista_3/sub_7:z:0lista_3/sub_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm_4/mul
$lista_3/norm_4/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2&
$lista_3/norm_4/Sum/reduction_indicesО
lista_3/norm_4/SumSumlista_3/norm_4/mul:z:0-lista_3/norm_4/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2
lista_3/norm_4/Sum
lista_3/norm_4/SqrtSqrtlista_3/norm_4/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm_4/Sqrt
lista_3/norm_4/SqueezeSqueezelista_3/norm_4/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
lista_3/norm_4/Squeezev
lista_3/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lista_3/ExpandDims_4/dimЕ
lista_3/ExpandDims_4
ExpandDimslista_3/norm_4/Squeeze:output:0!lista_3/ExpandDims_4/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/ExpandDims_4
lista_3/Tile_4/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
lista_3/Tile_4/multiplesЁ
lista_3/Tile_4Tilelista_3/ExpandDims_4:output:0!lista_3/Tile_4/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Tile_4І
 lista_3/Maximum_4/ReadVariableOpReadVariableOp)lista_3_maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02"
 lista_3/Maximum_4/ReadVariableOpЋ
lista_3/Maximum_4Maximumlista_3/Tile_4:output:0(lista_3/Maximum_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Maximum_4
lista_3/ReadVariableOp_4ReadVariableOp)lista_3_maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02
lista_3/ReadVariableOp_4Ё
lista_3/truediv_4RealDiv lista_3/ReadVariableOp_4:value:0lista_3/Maximum_4:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/truediv_4g
lista_3/sub_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lista_3/sub_8/x
lista_3/sub_8Sublista_3/sub_8/x:output:0lista_3/truediv_4:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/sub_8
lista_3/mul_4Mullista_3/sub_8:z:0lista_3/sub_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/mul_4И
#lista_3/Tensordot_14/ReadVariableOpReadVariableOp+lista_3_tensordot_2_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02%
#lista_3/Tensordot_14/ReadVariableOp
lista_3/Tensordot_14/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_14/axes
lista_3/Tensordot_14/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_14/freey
lista_3/Tensordot_14/ShapeShapelista_3/mul_3:z:0*
T0*
_output_shapes
:2
lista_3/Tensordot_14/Shape
"lista_3/Tensordot_14/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_14/GatherV2/axis
lista_3/Tensordot_14/GatherV2GatherV2#lista_3/Tensordot_14/Shape:output:0"lista_3/Tensordot_14/free:output:0+lista_3/Tensordot_14/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_14/GatherV2
$lista_3/Tensordot_14/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lista_3/Tensordot_14/GatherV2_1/axis
lista_3/Tensordot_14/GatherV2_1GatherV2#lista_3/Tensordot_14/Shape:output:0"lista_3/Tensordot_14/axes:output:0-lista_3/Tensordot_14/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
lista_3/Tensordot_14/GatherV2_1
lista_3/Tensordot_14/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_14/ConstЌ
lista_3/Tensordot_14/ProdProd&lista_3/Tensordot_14/GatherV2:output:0#lista_3/Tensordot_14/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_14/Prod
lista_3/Tensordot_14/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_14/Const_1Д
lista_3/Tensordot_14/Prod_1Prod(lista_3/Tensordot_14/GatherV2_1:output:0%lista_3/Tensordot_14/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_14/Prod_1
 lista_3/Tensordot_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lista_3/Tensordot_14/concat/axisч
lista_3/Tensordot_14/concatConcatV2"lista_3/Tensordot_14/axes:output:0"lista_3/Tensordot_14/free:output:0)lista_3/Tensordot_14/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_14/concatИ
lista_3/Tensordot_14/stackPack$lista_3/Tensordot_14/Prod_1:output:0"lista_3/Tensordot_14/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_14/stackН
lista_3/Tensordot_14/transpose	Transposelista_3/mul_3:z:0$lista_3/Tensordot_14/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2 
lista_3/Tensordot_14/transposeЫ
lista_3/Tensordot_14/ReshapeReshape"lista_3/Tensordot_14/transpose:y:0#lista_3/Tensordot_14/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_14/ReshapeЫ
lista_3/Tensordot_14/MatMulMatMul+lista_3/Tensordot_14/ReadVariableOp:value:0%lista_3/Tensordot_14/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_14/MatMul
lista_3/Tensordot_14/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_14/Const_2
"lista_3/Tensordot_14/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_14/concat_1/axisє
lista_3/Tensordot_14/concat_1ConcatV2%lista_3/Tensordot_14/Const_2:output:0&lista_3/Tensordot_14/GatherV2:output:0+lista_3/Tensordot_14/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_14/concat_1Н
lista_3/Tensordot_14Reshape%lista_3/Tensordot_14/MatMul:product:0&lista_3/Tensordot_14/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_14
lista_3/transpose_14/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_14/permГ
lista_3/transpose_14	Transposelista_3/Tensordot_14:output:0"lista_3/transpose_14/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_14И
#lista_3/Tensordot_15/ReadVariableOpReadVariableOp+lista_3_tensordot_3_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02%
#lista_3/Tensordot_15/ReadVariableOp
lista_3/Tensordot_15/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_15/axes
lista_3/Tensordot_15/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_15/freey
lista_3/Tensordot_15/ShapeShapelista_3/mul_4:z:0*
T0*
_output_shapes
:2
lista_3/Tensordot_15/Shape
"lista_3/Tensordot_15/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_15/GatherV2/axis
lista_3/Tensordot_15/GatherV2GatherV2#lista_3/Tensordot_15/Shape:output:0"lista_3/Tensordot_15/free:output:0+lista_3/Tensordot_15/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_15/GatherV2
$lista_3/Tensordot_15/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lista_3/Tensordot_15/GatherV2_1/axis
lista_3/Tensordot_15/GatherV2_1GatherV2#lista_3/Tensordot_15/Shape:output:0"lista_3/Tensordot_15/axes:output:0-lista_3/Tensordot_15/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
lista_3/Tensordot_15/GatherV2_1
lista_3/Tensordot_15/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_15/ConstЌ
lista_3/Tensordot_15/ProdProd&lista_3/Tensordot_15/GatherV2:output:0#lista_3/Tensordot_15/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_15/Prod
lista_3/Tensordot_15/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_15/Const_1Д
lista_3/Tensordot_15/Prod_1Prod(lista_3/Tensordot_15/GatherV2_1:output:0%lista_3/Tensordot_15/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_15/Prod_1
 lista_3/Tensordot_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lista_3/Tensordot_15/concat/axisч
lista_3/Tensordot_15/concatConcatV2"lista_3/Tensordot_15/axes:output:0"lista_3/Tensordot_15/free:output:0)lista_3/Tensordot_15/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_15/concatИ
lista_3/Tensordot_15/stackPack$lista_3/Tensordot_15/Prod_1:output:0"lista_3/Tensordot_15/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_15/stackН
lista_3/Tensordot_15/transpose	Transposelista_3/mul_4:z:0$lista_3/Tensordot_15/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2 
lista_3/Tensordot_15/transposeЫ
lista_3/Tensordot_15/ReshapeReshape"lista_3/Tensordot_15/transpose:y:0#lista_3/Tensordot_15/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_15/ReshapeЫ
lista_3/Tensordot_15/MatMulMatMul+lista_3/Tensordot_15/ReadVariableOp:value:0%lista_3/Tensordot_15/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_15/MatMul
lista_3/Tensordot_15/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_15/Const_2
"lista_3/Tensordot_15/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_15/concat_1/axisє
lista_3/Tensordot_15/concat_1ConcatV2%lista_3/Tensordot_15/Const_2:output:0&lista_3/Tensordot_15/GatherV2:output:0+lista_3/Tensordot_15/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_15/concat_1Н
lista_3/Tensordot_15Reshape%lista_3/Tensordot_15/MatMul:product:0&lista_3/Tensordot_15/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_15
lista_3/transpose_15/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_15/permГ
lista_3/transpose_15	Transposelista_3/Tensordot_15:output:0"lista_3/transpose_15/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_15
lista_3/add_5AddV2lista_3/transpose_14:y:0lista_3/transpose_15:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/add_5Е
#lista_3/Tensordot_16/ReadVariableOpReadVariableOp)lista_3_tensordot_readvariableop_resource*
_output_shapes
:	Д@*
dtype02%
#lista_3/Tensordot_16/ReadVariableOp
lista_3/Tensordot_16/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_16/axes
lista_3/Tensordot_16/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_16/free}
lista_3/Tensordot_16/ShapeShapelista_3/Imag:output:0*
T0*
_output_shapes
:2
lista_3/Tensordot_16/Shape
"lista_3/Tensordot_16/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_16/GatherV2/axis
lista_3/Tensordot_16/GatherV2GatherV2#lista_3/Tensordot_16/Shape:output:0"lista_3/Tensordot_16/free:output:0+lista_3/Tensordot_16/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_16/GatherV2
$lista_3/Tensordot_16/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lista_3/Tensordot_16/GatherV2_1/axis
lista_3/Tensordot_16/GatherV2_1GatherV2#lista_3/Tensordot_16/Shape:output:0"lista_3/Tensordot_16/axes:output:0-lista_3/Tensordot_16/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
lista_3/Tensordot_16/GatherV2_1
lista_3/Tensordot_16/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_16/ConstЌ
lista_3/Tensordot_16/ProdProd&lista_3/Tensordot_16/GatherV2:output:0#lista_3/Tensordot_16/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_16/Prod
lista_3/Tensordot_16/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_16/Const_1Д
lista_3/Tensordot_16/Prod_1Prod(lista_3/Tensordot_16/GatherV2_1:output:0%lista_3/Tensordot_16/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_16/Prod_1
 lista_3/Tensordot_16/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lista_3/Tensordot_16/concat/axisч
lista_3/Tensordot_16/concatConcatV2"lista_3/Tensordot_16/axes:output:0"lista_3/Tensordot_16/free:output:0)lista_3/Tensordot_16/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_16/concatИ
lista_3/Tensordot_16/stackPack$lista_3/Tensordot_16/Prod_1:output:0"lista_3/Tensordot_16/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_16/stackР
lista_3/Tensordot_16/transpose	Transposelista_3/Imag:output:0$lista_3/Tensordot_16/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2 
lista_3/Tensordot_16/transposeЫ
lista_3/Tensordot_16/ReshapeReshape"lista_3/Tensordot_16/transpose:y:0#lista_3/Tensordot_16/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_16/ReshapeЫ
lista_3/Tensordot_16/MatMulMatMul+lista_3/Tensordot_16/ReadVariableOp:value:0%lista_3/Tensordot_16/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_16/MatMul
lista_3/Tensordot_16/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_16/Const_2
"lista_3/Tensordot_16/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_16/concat_1/axisє
lista_3/Tensordot_16/concat_1ConcatV2%lista_3/Tensordot_16/Const_2:output:0&lista_3/Tensordot_16/GatherV2:output:0+lista_3/Tensordot_16/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_16/concat_1Н
lista_3/Tensordot_16Reshape%lista_3/Tensordot_16/MatMul:product:0&lista_3/Tensordot_16/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_16
lista_3/transpose_16/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_16/permГ
lista_3/transpose_16	Transposelista_3/Tensordot_16:output:0"lista_3/transpose_16/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_16
lista_3/add_6AddV2lista_3/add_5:z:0lista_3/transpose_16:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/add_6З
#lista_3/Tensordot_17/ReadVariableOpReadVariableOp+lista_3_tensordot_1_readvariableop_resource*
_output_shapes
:	Д@*
dtype02%
#lista_3/Tensordot_17/ReadVariableOp
lista_3/Tensordot_17/axesConst*
_output_shapes
:*
dtype0*
valueB:2
lista_3/Tensordot_17/axes
lista_3/Tensordot_17/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
lista_3/Tensordot_17/free}
lista_3/Tensordot_17/ShapeShapelista_3/Real:output:0*
T0*
_output_shapes
:2
lista_3/Tensordot_17/Shape
"lista_3/Tensordot_17/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_17/GatherV2/axis
lista_3/Tensordot_17/GatherV2GatherV2#lista_3/Tensordot_17/Shape:output:0"lista_3/Tensordot_17/free:output:0+lista_3/Tensordot_17/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
lista_3/Tensordot_17/GatherV2
$lista_3/Tensordot_17/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lista_3/Tensordot_17/GatherV2_1/axis
lista_3/Tensordot_17/GatherV2_1GatherV2#lista_3/Tensordot_17/Shape:output:0"lista_3/Tensordot_17/axes:output:0-lista_3/Tensordot_17/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
lista_3/Tensordot_17/GatherV2_1
lista_3/Tensordot_17/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_17/ConstЌ
lista_3/Tensordot_17/ProdProd&lista_3/Tensordot_17/GatherV2:output:0#lista_3/Tensordot_17/Const:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_17/Prod
lista_3/Tensordot_17/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lista_3/Tensordot_17/Const_1Д
lista_3/Tensordot_17/Prod_1Prod(lista_3/Tensordot_17/GatherV2_1:output:0%lista_3/Tensordot_17/Const_1:output:0*
T0*
_output_shapes
: 2
lista_3/Tensordot_17/Prod_1
 lista_3/Tensordot_17/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lista_3/Tensordot_17/concat/axisч
lista_3/Tensordot_17/concatConcatV2"lista_3/Tensordot_17/axes:output:0"lista_3/Tensordot_17/free:output:0)lista_3/Tensordot_17/concat/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_17/concatИ
lista_3/Tensordot_17/stackPack$lista_3/Tensordot_17/Prod_1:output:0"lista_3/Tensordot_17/Prod:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_17/stackР
lista_3/Tensordot_17/transpose	Transposelista_3/Real:output:0$lista_3/Tensordot_17/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2 
lista_3/Tensordot_17/transposeЫ
lista_3/Tensordot_17/ReshapeReshape"lista_3/Tensordot_17/transpose:y:0#lista_3/Tensordot_17/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
lista_3/Tensordot_17/ReshapeЫ
lista_3/Tensordot_17/MatMulMatMul+lista_3/Tensordot_17/ReadVariableOp:value:0%lista_3/Tensordot_17/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_17/MatMul
lista_3/Tensordot_17/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
lista_3/Tensordot_17/Const_2
"lista_3/Tensordot_17/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lista_3/Tensordot_17/concat_1/axisє
lista_3/Tensordot_17/concat_1ConcatV2%lista_3/Tensordot_17/Const_2:output:0&lista_3/Tensordot_17/GatherV2:output:0+lista_3/Tensordot_17/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
lista_3/Tensordot_17/concat_1Н
lista_3/Tensordot_17Reshape%lista_3/Tensordot_17/MatMul:product:0&lista_3/Tensordot_17/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
lista_3/Tensordot_17
lista_3/transpose_17/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lista_3/transpose_17/permГ
lista_3/transpose_17	Transposelista_3/Tensordot_17:output:0"lista_3/transpose_17/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/transpose_17
lista_3/add_7AddV2lista_3/add_6:z:0lista_3/transpose_17:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/add_7
lista_3/norm_5/mulMullista_3/add_7:z:0lista_3/add_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm_5/mul
$lista_3/norm_5/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2&
$lista_3/norm_5/Sum/reduction_indicesО
lista_3/norm_5/SumSumlista_3/norm_5/mul:z:0-lista_3/norm_5/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2
lista_3/norm_5/Sum
lista_3/norm_5/SqrtSqrtlista_3/norm_5/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/norm_5/Sqrt
lista_3/norm_5/SqueezeSqueezelista_3/norm_5/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
lista_3/norm_5/Squeezev
lista_3/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lista_3/ExpandDims_5/dimЕ
lista_3/ExpandDims_5
ExpandDimslista_3/norm_5/Squeeze:output:0!lista_3/ExpandDims_5/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/ExpandDims_5
lista_3/Tile_5/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
lista_3/Tile_5/multiplesЁ
lista_3/Tile_5Tilelista_3/ExpandDims_5:output:0!lista_3/Tile_5/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Tile_5І
 lista_3/Maximum_5/ReadVariableOpReadVariableOp)lista_3_maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02"
 lista_3/Maximum_5/ReadVariableOpЋ
lista_3/Maximum_5Maximumlista_3/Tile_5:output:0(lista_3/Maximum_5/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/Maximum_5
lista_3/ReadVariableOp_5ReadVariableOp)lista_3_maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02
lista_3/ReadVariableOp_5Ё
lista_3/truediv_5RealDiv lista_3/ReadVariableOp_5:value:0lista_3/Maximum_5:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/truediv_5g
lista_3/sub_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lista_3/sub_9/x
lista_3/sub_9Sublista_3/sub_9/x:output:0lista_3/truediv_5:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/sub_9
lista_3/mul_5Mullista_3/sub_9:z:0lista_3/add_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
lista_3/mul_5l
lista_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
lista_3/concat/axisА
lista_3/concatConcatV2lista_3/mul_4:z:0lista_3/mul_5:z:0lista_3/concat/axis:output:0*
N*
T0*,
_output_shapes
:џџџџџџџџџш2
lista_3/concatp
IdentityIdentitylista_3/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџ@::::::::T P
+
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
Еф
ј
C__inference_lista_3_layer_call_and_return_conditional_losses_903569
input_1%
!tensordot_readvariableop_resource#
maximum_readvariableop_resource'
#tensordot_1_readvariableop_resource'
#tensordot_2_readvariableop_resource'
#tensordot_3_readvariableop_resource%
!maximum_2_readvariableop_resource%
!maximum_4_readvariableop_resource
identityJ
RealRealinput_1*+
_output_shapes
:џџџџџџџџџ@2
RealJ
ImagImaginput_1*+
_output_shapes
:џџџџџџџџџ@2
Imag
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Д@*
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
Tensordot/GatherV2/axisб
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
Tensordot/GatherV2_1/axisз
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/axes:output:0Tensordot/free:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod_1:output:0Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeReal:output:0Tensordot/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMul Tensordot/ReadVariableOp:value:0Tensordot/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/Const_2:output:0Tensordot/GatherV2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
	Tensordotu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	TransposeTensordot:output:0transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	transposep
norm/mulMultranspose:y:0transpose:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2

norm/mul
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm/Sum/reduction_indices
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2

norm/Sumh
	norm/SqrtSqrtnorm/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	norm/Sqrt
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
norm/Squeezeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsnorm/Squeeze:output:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2

ExpandDimsu
Tile/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile/multiplesy
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
Tile
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum/ReadVariableOp
MaximumMaximumTile:output:0Maximum/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2	
Maximumx
ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpy
truedivRealDivReadVariableOp:value:0Maximum:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xe
subSubsub/x:output:0truediv:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
sub`
mulMulsub:z:0transpose:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
mul
Tensordot_1/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource*
_output_shapes
:	Д@*
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
Tensordot_1/GatherV2/axisл
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
Tensordot_1/GatherV2_1/axisс
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
Tensordot_1/Const
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
Tensordot_1/Const_1
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
Tensordot_1/concat/axisК
Tensordot_1/concatConcatV2Tensordot_1/axes:output:0Tensordot_1/free:output:0 Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_1/concat
Tensordot_1/stackPackTensordot_1/Prod_1:output:0Tensordot_1/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_1/stack
Tensordot_1/transpose	TransposeImag:output:0Tensordot_1/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
Tensordot_1/transposeЇ
Tensordot_1/ReshapeReshapeTensordot_1/transpose:y:0Tensordot_1/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_1/ReshapeЇ
Tensordot_1/MatMulMatMul"Tensordot_1/ReadVariableOp:value:0Tensordot_1/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_1/MatMulu
Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_1/Const_2x
Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_1/concat_1/axisЧ
Tensordot_1/concat_1ConcatV2Tensordot_1/Const_2:output:0Tensordot_1/GatherV2:output:0"Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_1/concat_1
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm
transpose_1	TransposeTensordot_1:output:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_1x

norm_1/mulMultranspose_1:y:0transpose_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2

norm_1/mul
norm_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm_1/Sum/reduction_indices

norm_1/SumSumnorm_1/mul:z:0%norm_1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2

norm_1/Sumn
norm_1/SqrtSqrtnorm_1/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
norm_1/Sqrt
norm_1/SqueezeSqueezenorm_1/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
norm_1/Squeezef
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsnorm_1/Squeeze:output:0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
ExpandDims_1y
Tile_1/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile_1/multiples
Tile_1TileExpandDims_1:output:0Tile_1/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
Tile_1
Maximum_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum_1/ReadVariableOp
	Maximum_1MaximumTile_1:output:0 Maximum_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	Maximum_1|
ReadVariableOp_1ReadVariableOpmaximum_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1
	truediv_1RealDivReadVariableOp_1:value:0Maximum_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	truediv_1W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/xm
sub_1Subsub_1/x:output:0truediv_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
sub_1h
mul_1Mul	sub_1:z:0transpose_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
mul_1
Tensordot_2/ReadVariableOpReadVariableOp#tensordot_2_readvariableop_resource* 
_output_shapes
:
ДД*
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
Tensordot_2/free]
Tensordot_2/ShapeShapemul:z:0*
T0*
_output_shapes
:2
Tensordot_2/Shapex
Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_2/GatherV2/axisл
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
Tensordot_2/GatherV2_1/axisс
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
Tensordot_2/Const
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
Tensordot_2/Const_1
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
Tensordot_2/concat/axisК
Tensordot_2/concatConcatV2Tensordot_2/axes:output:0Tensordot_2/free:output:0 Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_2/concat
Tensordot_2/stackPackTensordot_2/Prod_1:output:0Tensordot_2/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_2/stack
Tensordot_2/transpose	Transposemul:z:0Tensordot_2/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_2/transposeЇ
Tensordot_2/ReshapeReshapeTensordot_2/transpose:y:0Tensordot_2/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_2/ReshapeЇ
Tensordot_2/MatMulMatMul"Tensordot_2/ReadVariableOp:value:0Tensordot_2/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_2/MatMulu
Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_2/Const_2x
Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_2/concat_1/axisЧ
Tensordot_2/concat_1ConcatV2Tensordot_2/Const_2:output:0Tensordot_2/GatherV2:output:0"Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_2/concat_1
Tensordot_2ReshapeTensordot_2/MatMul:product:0Tensordot_2/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeTensordot_2:output:0transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_2
Tensordot_3/ReadVariableOpReadVariableOp#tensordot_3_readvariableop_resource* 
_output_shapes
:
ДД*
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
Tensordot_3/free_
Tensordot_3/ShapeShape	mul_1:z:0*
T0*
_output_shapes
:2
Tensordot_3/Shapex
Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_3/GatherV2/axisл
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
Tensordot_3/GatherV2_1/axisс
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
Tensordot_3/Const
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
Tensordot_3/Const_1
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
Tensordot_3/concat/axisК
Tensordot_3/concatConcatV2Tensordot_3/axes:output:0Tensordot_3/free:output:0 Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_3/concat
Tensordot_3/stackPackTensordot_3/Prod_1:output:0Tensordot_3/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_3/stack
Tensordot_3/transpose	Transpose	mul_1:z:0Tensordot_3/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_3/transposeЇ
Tensordot_3/ReshapeReshapeTensordot_3/transpose:y:0Tensordot_3/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_3/ReshapeЇ
Tensordot_3/MatMulMatMul"Tensordot_3/ReadVariableOp:value:0Tensordot_3/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_3/MatMulu
Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_3/Const_2x
Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_3/concat_1/axisЧ
Tensordot_3/concat_1ConcatV2Tensordot_3/Const_2:output:0Tensordot_3/GatherV2:output:0"Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_3/concat_1
Tensordot_3ReshapeTensordot_3/MatMul:product:0Tensordot_3/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_3y
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_3/perm
transpose_3	TransposeTensordot_3:output:0transpose_3/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_3n
sub_2Subtranspose_2:y:0transpose_3:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
sub_2
Tensordot_4/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Д@*
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
Tensordot_4/ShapeShapeReal:output:0*
T0*
_output_shapes
:2
Tensordot_4/Shapex
Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_4/GatherV2/axisл
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
Tensordot_4/GatherV2_1/axisс
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
Tensordot_4/Const
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
Tensordot_4/Const_1
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
Tensordot_4/concat/axisК
Tensordot_4/concatConcatV2Tensordot_4/axes:output:0Tensordot_4/free:output:0 Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_4/concat
Tensordot_4/stackPackTensordot_4/Prod_1:output:0Tensordot_4/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_4/stack
Tensordot_4/transpose	TransposeReal:output:0Tensordot_4/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
Tensordot_4/transposeЇ
Tensordot_4/ReshapeReshapeTensordot_4/transpose:y:0Tensordot_4/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_4/ReshapeЇ
Tensordot_4/MatMulMatMul"Tensordot_4/ReadVariableOp:value:0Tensordot_4/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_4/MatMulu
Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_4/Const_2x
Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_4/concat_1/axisЧ
Tensordot_4/concat_1ConcatV2Tensordot_4/Const_2:output:0Tensordot_4/GatherV2:output:0"Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_4/concat_1
Tensordot_4ReshapeTensordot_4/MatMul:product:0Tensordot_4/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_4y
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_4/perm
transpose_4	TransposeTensordot_4:output:0transpose_4/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_4f
addAddV2	sub_2:z:0transpose_4:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
add
Tensordot_5/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource*
_output_shapes
:	Д@*
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
Tensordot_5/ShapeShapeImag:output:0*
T0*
_output_shapes
:2
Tensordot_5/Shapex
Tensordot_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_5/GatherV2/axisл
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
Tensordot_5/GatherV2_1/axisс
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
Tensordot_5/Const
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
Tensordot_5/Const_1
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
Tensordot_5/concat/axisК
Tensordot_5/concatConcatV2Tensordot_5/axes:output:0Tensordot_5/free:output:0 Tensordot_5/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_5/concat
Tensordot_5/stackPackTensordot_5/Prod_1:output:0Tensordot_5/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_5/stack
Tensordot_5/transpose	TransposeImag:output:0Tensordot_5/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
Tensordot_5/transposeЇ
Tensordot_5/ReshapeReshapeTensordot_5/transpose:y:0Tensordot_5/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_5/ReshapeЇ
Tensordot_5/MatMulMatMul"Tensordot_5/ReadVariableOp:value:0Tensordot_5/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_5/MatMulu
Tensordot_5/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_5/Const_2x
Tensordot_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_5/concat_1/axisЧ
Tensordot_5/concat_1ConcatV2Tensordot_5/Const_2:output:0Tensordot_5/GatherV2:output:0"Tensordot_5/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_5/concat_1
Tensordot_5ReshapeTensordot_5/MatMul:product:0Tensordot_5/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_5y
transpose_5/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_5/perm
transpose_5	TransposeTensordot_5:output:0transpose_5/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_5f
sub_3Subadd:z:0transpose_5:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
sub_3l

norm_2/mulMul	sub_3:z:0	sub_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2

norm_2/mul
norm_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm_2/Sum/reduction_indices

norm_2/SumSumnorm_2/mul:z:0%norm_2/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2

norm_2/Sumn
norm_2/SqrtSqrtnorm_2/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
norm_2/Sqrt
norm_2/SqueezeSqueezenorm_2/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
norm_2/Squeezef
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_2/dim
ExpandDims_2
ExpandDimsnorm_2/Squeeze:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
ExpandDims_2y
Tile_2/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile_2/multiples
Tile_2TileExpandDims_2:output:0Tile_2/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
Tile_2
Maximum_2/ReadVariableOpReadVariableOp!maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum_2/ReadVariableOp
	Maximum_2MaximumTile_2:output:0 Maximum_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	Maximum_2~
ReadVariableOp_2ReadVariableOp!maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2
	truediv_2RealDivReadVariableOp_2:value:0Maximum_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	truediv_2W
sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_4/xm
sub_4Subsub_4/x:output:0truediv_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
sub_4b
mul_2Mul	sub_4:z:0	sub_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
mul_2
Tensordot_6/ReadVariableOpReadVariableOp#tensordot_2_readvariableop_resource* 
_output_shapes
:
ДД*
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
Tensordot_6/free_
Tensordot_6/ShapeShape	mul_1:z:0*
T0*
_output_shapes
:2
Tensordot_6/Shapex
Tensordot_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_6/GatherV2/axisл
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
Tensordot_6/GatherV2_1/axisс
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
Tensordot_6/Const
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
Tensordot_6/Const_1
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
Tensordot_6/concat/axisК
Tensordot_6/concatConcatV2Tensordot_6/axes:output:0Tensordot_6/free:output:0 Tensordot_6/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_6/concat
Tensordot_6/stackPackTensordot_6/Prod_1:output:0Tensordot_6/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_6/stack
Tensordot_6/transpose	Transpose	mul_1:z:0Tensordot_6/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_6/transposeЇ
Tensordot_6/ReshapeReshapeTensordot_6/transpose:y:0Tensordot_6/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_6/ReshapeЇ
Tensordot_6/MatMulMatMul"Tensordot_6/ReadVariableOp:value:0Tensordot_6/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_6/MatMulu
Tensordot_6/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_6/Const_2x
Tensordot_6/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_6/concat_1/axisЧ
Tensordot_6/concat_1ConcatV2Tensordot_6/Const_2:output:0Tensordot_6/GatherV2:output:0"Tensordot_6/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_6/concat_1
Tensordot_6ReshapeTensordot_6/MatMul:product:0Tensordot_6/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_6y
transpose_6/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_6/perm
transpose_6	TransposeTensordot_6:output:0transpose_6/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_6
Tensordot_7/ReadVariableOpReadVariableOp#tensordot_3_readvariableop_resource* 
_output_shapes
:
ДД*
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
Tensordot_7/free_
Tensordot_7/ShapeShape	mul_2:z:0*
T0*
_output_shapes
:2
Tensordot_7/Shapex
Tensordot_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_7/GatherV2/axisл
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
Tensordot_7/GatherV2_1/axisс
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
Tensordot_7/Const
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
Tensordot_7/Const_1
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
Tensordot_7/concat/axisК
Tensordot_7/concatConcatV2Tensordot_7/axes:output:0Tensordot_7/free:output:0 Tensordot_7/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_7/concat
Tensordot_7/stackPackTensordot_7/Prod_1:output:0Tensordot_7/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_7/stack
Tensordot_7/transpose	Transpose	mul_2:z:0Tensordot_7/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_7/transposeЇ
Tensordot_7/ReshapeReshapeTensordot_7/transpose:y:0Tensordot_7/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_7/ReshapeЇ
Tensordot_7/MatMulMatMul"Tensordot_7/ReadVariableOp:value:0Tensordot_7/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_7/MatMulu
Tensordot_7/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_7/Const_2x
Tensordot_7/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_7/concat_1/axisЧ
Tensordot_7/concat_1ConcatV2Tensordot_7/Const_2:output:0Tensordot_7/GatherV2:output:0"Tensordot_7/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_7/concat_1
Tensordot_7ReshapeTensordot_7/MatMul:product:0Tensordot_7/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_7y
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeTensordot_7:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_7p
add_1AddV2transpose_6:y:0transpose_7:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
add_1
Tensordot_8/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Д@*
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
Tensordot_8/GatherV2/axisл
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
Tensordot_8/GatherV2_1/axisс
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
Tensordot_8/Const
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
Tensordot_8/Const_1
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
Tensordot_8/concat/axisК
Tensordot_8/concatConcatV2Tensordot_8/axes:output:0Tensordot_8/free:output:0 Tensordot_8/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_8/concat
Tensordot_8/stackPackTensordot_8/Prod_1:output:0Tensordot_8/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_8/stack
Tensordot_8/transpose	TransposeImag:output:0Tensordot_8/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
Tensordot_8/transposeЇ
Tensordot_8/ReshapeReshapeTensordot_8/transpose:y:0Tensordot_8/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_8/ReshapeЇ
Tensordot_8/MatMulMatMul"Tensordot_8/ReadVariableOp:value:0Tensordot_8/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_8/MatMulu
Tensordot_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_8/Const_2x
Tensordot_8/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_8/concat_1/axisЧ
Tensordot_8/concat_1ConcatV2Tensordot_8/Const_2:output:0Tensordot_8/GatherV2:output:0"Tensordot_8/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_8/concat_1
Tensordot_8ReshapeTensordot_8/MatMul:product:0Tensordot_8/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_8y
transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_8/perm
transpose_8	TransposeTensordot_8:output:0transpose_8/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_8j
add_2AddV2	add_1:z:0transpose_8:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
add_2
Tensordot_9/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource*
_output_shapes
:	Д@*
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
Tensordot_9/GatherV2/axisл
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
Tensordot_9/GatherV2_1/axisс
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
Tensordot_9/Const
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
Tensordot_9/Const_1
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
Tensordot_9/concat/axisК
Tensordot_9/concatConcatV2Tensordot_9/axes:output:0Tensordot_9/free:output:0 Tensordot_9/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_9/concat
Tensordot_9/stackPackTensordot_9/Prod_1:output:0Tensordot_9/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_9/stack
Tensordot_9/transpose	TransposeReal:output:0Tensordot_9/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
Tensordot_9/transposeЇ
Tensordot_9/ReshapeReshapeTensordot_9/transpose:y:0Tensordot_9/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_9/ReshapeЇ
Tensordot_9/MatMulMatMul"Tensordot_9/ReadVariableOp:value:0Tensordot_9/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_9/MatMulu
Tensordot_9/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_9/Const_2x
Tensordot_9/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_9/concat_1/axisЧ
Tensordot_9/concat_1ConcatV2Tensordot_9/Const_2:output:0Tensordot_9/GatherV2:output:0"Tensordot_9/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_9/concat_1
Tensordot_9ReshapeTensordot_9/MatMul:product:0Tensordot_9/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_9y
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeTensordot_9:output:0transpose_9/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_9j
add_3AddV2	add_2:z:0transpose_9:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
add_3l

norm_3/mulMul	add_3:z:0	add_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2

norm_3/mul
norm_3/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm_3/Sum/reduction_indices

norm_3/SumSumnorm_3/mul:z:0%norm_3/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2

norm_3/Sumn
norm_3/SqrtSqrtnorm_3/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
norm_3/Sqrt
norm_3/SqueezeSqueezenorm_3/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
norm_3/Squeezef
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_3/dim
ExpandDims_3
ExpandDimsnorm_3/Squeeze:output:0ExpandDims_3/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
ExpandDims_3y
Tile_3/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile_3/multiples
Tile_3TileExpandDims_3:output:0Tile_3/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
Tile_3
Maximum_3/ReadVariableOpReadVariableOp!maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum_3/ReadVariableOp
	Maximum_3MaximumTile_3:output:0 Maximum_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	Maximum_3~
ReadVariableOp_3ReadVariableOp!maximum_2_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_3
	truediv_3RealDivReadVariableOp_3:value:0Maximum_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	truediv_3W
sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_5/xm
sub_5Subsub_5/x:output:0truediv_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
sub_5b
mul_3Mul	sub_5:z:0	add_3:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
mul_3 
Tensordot_10/ReadVariableOpReadVariableOp#tensordot_2_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02
Tensordot_10/ReadVariableOpp
Tensordot_10/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_10/axesw
Tensordot_10/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_10/freea
Tensordot_10/ShapeShape	mul_2:z:0*
T0*
_output_shapes
:2
Tensordot_10/Shapez
Tensordot_10/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_10/GatherV2/axisр
Tensordot_10/GatherV2GatherV2Tensordot_10/Shape:output:0Tensordot_10/free:output:0#Tensordot_10/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_10/GatherV2~
Tensordot_10/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_10/GatherV2_1/axisц
Tensordot_10/GatherV2_1GatherV2Tensordot_10/Shape:output:0Tensordot_10/axes:output:0%Tensordot_10/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_10/GatherV2_1r
Tensordot_10/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_10/Const
Tensordot_10/ProdProdTensordot_10/GatherV2:output:0Tensordot_10/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_10/Prodv
Tensordot_10/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_10/Const_1
Tensordot_10/Prod_1Prod Tensordot_10/GatherV2_1:output:0Tensordot_10/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_10/Prod_1v
Tensordot_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_10/concat/axisП
Tensordot_10/concatConcatV2Tensordot_10/axes:output:0Tensordot_10/free:output:0!Tensordot_10/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_10/concat
Tensordot_10/stackPackTensordot_10/Prod_1:output:0Tensordot_10/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_10/stack
Tensordot_10/transpose	Transpose	mul_2:z:0Tensordot_10/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_10/transposeЋ
Tensordot_10/ReshapeReshapeTensordot_10/transpose:y:0Tensordot_10/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_10/ReshapeЋ
Tensordot_10/MatMulMatMul#Tensordot_10/ReadVariableOp:value:0Tensordot_10/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_10/MatMulw
Tensordot_10/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_10/Const_2z
Tensordot_10/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_10/concat_1/axisЬ
Tensordot_10/concat_1ConcatV2Tensordot_10/Const_2:output:0Tensordot_10/GatherV2:output:0#Tensordot_10/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_10/concat_1
Tensordot_10ReshapeTensordot_10/MatMul:product:0Tensordot_10/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_10{
transpose_10/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_10/perm
transpose_10	TransposeTensordot_10:output:0transpose_10/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_10 
Tensordot_11/ReadVariableOpReadVariableOp#tensordot_3_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02
Tensordot_11/ReadVariableOpp
Tensordot_11/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_11/axesw
Tensordot_11/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_11/freea
Tensordot_11/ShapeShape	mul_3:z:0*
T0*
_output_shapes
:2
Tensordot_11/Shapez
Tensordot_11/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_11/GatherV2/axisр
Tensordot_11/GatherV2GatherV2Tensordot_11/Shape:output:0Tensordot_11/free:output:0#Tensordot_11/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_11/GatherV2~
Tensordot_11/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_11/GatherV2_1/axisц
Tensordot_11/GatherV2_1GatherV2Tensordot_11/Shape:output:0Tensordot_11/axes:output:0%Tensordot_11/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_11/GatherV2_1r
Tensordot_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_11/Const
Tensordot_11/ProdProdTensordot_11/GatherV2:output:0Tensordot_11/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_11/Prodv
Tensordot_11/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_11/Const_1
Tensordot_11/Prod_1Prod Tensordot_11/GatherV2_1:output:0Tensordot_11/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_11/Prod_1v
Tensordot_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_11/concat/axisП
Tensordot_11/concatConcatV2Tensordot_11/axes:output:0Tensordot_11/free:output:0!Tensordot_11/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_11/concat
Tensordot_11/stackPackTensordot_11/Prod_1:output:0Tensordot_11/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_11/stack
Tensordot_11/transpose	Transpose	mul_3:z:0Tensordot_11/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_11/transposeЋ
Tensordot_11/ReshapeReshapeTensordot_11/transpose:y:0Tensordot_11/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_11/ReshapeЋ
Tensordot_11/MatMulMatMul#Tensordot_11/ReadVariableOp:value:0Tensordot_11/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_11/MatMulw
Tensordot_11/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_11/Const_2z
Tensordot_11/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_11/concat_1/axisЬ
Tensordot_11/concat_1ConcatV2Tensordot_11/Const_2:output:0Tensordot_11/GatherV2:output:0#Tensordot_11/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_11/concat_1
Tensordot_11ReshapeTensordot_11/MatMul:product:0Tensordot_11/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_11{
transpose_11/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_11/perm
transpose_11	TransposeTensordot_11:output:0transpose_11/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_11p
sub_6Subtranspose_10:y:0transpose_11:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
sub_6
Tensordot_12/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Д@*
dtype02
Tensordot_12/ReadVariableOpp
Tensordot_12/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_12/axesw
Tensordot_12/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_12/freee
Tensordot_12/ShapeShapeReal:output:0*
T0*
_output_shapes
:2
Tensordot_12/Shapez
Tensordot_12/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_12/GatherV2/axisр
Tensordot_12/GatherV2GatherV2Tensordot_12/Shape:output:0Tensordot_12/free:output:0#Tensordot_12/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_12/GatherV2~
Tensordot_12/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_12/GatherV2_1/axisц
Tensordot_12/GatherV2_1GatherV2Tensordot_12/Shape:output:0Tensordot_12/axes:output:0%Tensordot_12/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_12/GatherV2_1r
Tensordot_12/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_12/Const
Tensordot_12/ProdProdTensordot_12/GatherV2:output:0Tensordot_12/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_12/Prodv
Tensordot_12/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_12/Const_1
Tensordot_12/Prod_1Prod Tensordot_12/GatherV2_1:output:0Tensordot_12/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_12/Prod_1v
Tensordot_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_12/concat/axisП
Tensordot_12/concatConcatV2Tensordot_12/axes:output:0Tensordot_12/free:output:0!Tensordot_12/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_12/concat
Tensordot_12/stackPackTensordot_12/Prod_1:output:0Tensordot_12/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_12/stack 
Tensordot_12/transpose	TransposeReal:output:0Tensordot_12/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
Tensordot_12/transposeЋ
Tensordot_12/ReshapeReshapeTensordot_12/transpose:y:0Tensordot_12/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_12/ReshapeЋ
Tensordot_12/MatMulMatMul#Tensordot_12/ReadVariableOp:value:0Tensordot_12/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_12/MatMulw
Tensordot_12/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_12/Const_2z
Tensordot_12/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_12/concat_1/axisЬ
Tensordot_12/concat_1ConcatV2Tensordot_12/Const_2:output:0Tensordot_12/GatherV2:output:0#Tensordot_12/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_12/concat_1
Tensordot_12ReshapeTensordot_12/MatMul:product:0Tensordot_12/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_12{
transpose_12/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_12/perm
transpose_12	TransposeTensordot_12:output:0transpose_12/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_12k
add_4AddV2	sub_6:z:0transpose_12:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
add_4
Tensordot_13/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource*
_output_shapes
:	Д@*
dtype02
Tensordot_13/ReadVariableOpp
Tensordot_13/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_13/axesw
Tensordot_13/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_13/freee
Tensordot_13/ShapeShapeImag:output:0*
T0*
_output_shapes
:2
Tensordot_13/Shapez
Tensordot_13/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_13/GatherV2/axisр
Tensordot_13/GatherV2GatherV2Tensordot_13/Shape:output:0Tensordot_13/free:output:0#Tensordot_13/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_13/GatherV2~
Tensordot_13/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_13/GatherV2_1/axisц
Tensordot_13/GatherV2_1GatherV2Tensordot_13/Shape:output:0Tensordot_13/axes:output:0%Tensordot_13/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_13/GatherV2_1r
Tensordot_13/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_13/Const
Tensordot_13/ProdProdTensordot_13/GatherV2:output:0Tensordot_13/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_13/Prodv
Tensordot_13/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_13/Const_1
Tensordot_13/Prod_1Prod Tensordot_13/GatherV2_1:output:0Tensordot_13/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_13/Prod_1v
Tensordot_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_13/concat/axisП
Tensordot_13/concatConcatV2Tensordot_13/axes:output:0Tensordot_13/free:output:0!Tensordot_13/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_13/concat
Tensordot_13/stackPackTensordot_13/Prod_1:output:0Tensordot_13/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_13/stack 
Tensordot_13/transpose	TransposeImag:output:0Tensordot_13/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
Tensordot_13/transposeЋ
Tensordot_13/ReshapeReshapeTensordot_13/transpose:y:0Tensordot_13/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_13/ReshapeЋ
Tensordot_13/MatMulMatMul#Tensordot_13/ReadVariableOp:value:0Tensordot_13/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_13/MatMulw
Tensordot_13/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_13/Const_2z
Tensordot_13/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_13/concat_1/axisЬ
Tensordot_13/concat_1ConcatV2Tensordot_13/Const_2:output:0Tensordot_13/GatherV2:output:0#Tensordot_13/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_13/concat_1
Tensordot_13ReshapeTensordot_13/MatMul:product:0Tensordot_13/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_13{
transpose_13/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_13/perm
transpose_13	TransposeTensordot_13:output:0transpose_13/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_13i
sub_7Sub	add_4:z:0transpose_13:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
sub_7l

norm_4/mulMul	sub_7:z:0	sub_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2

norm_4/mul
norm_4/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm_4/Sum/reduction_indices

norm_4/SumSumnorm_4/mul:z:0%norm_4/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2

norm_4/Sumn
norm_4/SqrtSqrtnorm_4/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
norm_4/Sqrt
norm_4/SqueezeSqueezenorm_4/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
norm_4/Squeezef
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_4/dim
ExpandDims_4
ExpandDimsnorm_4/Squeeze:output:0ExpandDims_4/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
ExpandDims_4y
Tile_4/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile_4/multiples
Tile_4TileExpandDims_4:output:0Tile_4/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
Tile_4
Maximum_4/ReadVariableOpReadVariableOp!maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum_4/ReadVariableOp
	Maximum_4MaximumTile_4:output:0 Maximum_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	Maximum_4~
ReadVariableOp_4ReadVariableOp!maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_4
	truediv_4RealDivReadVariableOp_4:value:0Maximum_4:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	truediv_4W
sub_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_8/xm
sub_8Subsub_8/x:output:0truediv_4:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
sub_8b
mul_4Mul	sub_8:z:0	sub_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
mul_4 
Tensordot_14/ReadVariableOpReadVariableOp#tensordot_2_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02
Tensordot_14/ReadVariableOpp
Tensordot_14/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_14/axesw
Tensordot_14/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_14/freea
Tensordot_14/ShapeShape	mul_3:z:0*
T0*
_output_shapes
:2
Tensordot_14/Shapez
Tensordot_14/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_14/GatherV2/axisр
Tensordot_14/GatherV2GatherV2Tensordot_14/Shape:output:0Tensordot_14/free:output:0#Tensordot_14/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_14/GatherV2~
Tensordot_14/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_14/GatherV2_1/axisц
Tensordot_14/GatherV2_1GatherV2Tensordot_14/Shape:output:0Tensordot_14/axes:output:0%Tensordot_14/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_14/GatherV2_1r
Tensordot_14/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_14/Const
Tensordot_14/ProdProdTensordot_14/GatherV2:output:0Tensordot_14/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_14/Prodv
Tensordot_14/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_14/Const_1
Tensordot_14/Prod_1Prod Tensordot_14/GatherV2_1:output:0Tensordot_14/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_14/Prod_1v
Tensordot_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_14/concat/axisП
Tensordot_14/concatConcatV2Tensordot_14/axes:output:0Tensordot_14/free:output:0!Tensordot_14/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_14/concat
Tensordot_14/stackPackTensordot_14/Prod_1:output:0Tensordot_14/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_14/stack
Tensordot_14/transpose	Transpose	mul_3:z:0Tensordot_14/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_14/transposeЋ
Tensordot_14/ReshapeReshapeTensordot_14/transpose:y:0Tensordot_14/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_14/ReshapeЋ
Tensordot_14/MatMulMatMul#Tensordot_14/ReadVariableOp:value:0Tensordot_14/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_14/MatMulw
Tensordot_14/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_14/Const_2z
Tensordot_14/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_14/concat_1/axisЬ
Tensordot_14/concat_1ConcatV2Tensordot_14/Const_2:output:0Tensordot_14/GatherV2:output:0#Tensordot_14/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_14/concat_1
Tensordot_14ReshapeTensordot_14/MatMul:product:0Tensordot_14/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_14{
transpose_14/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_14/perm
transpose_14	TransposeTensordot_14:output:0transpose_14/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_14 
Tensordot_15/ReadVariableOpReadVariableOp#tensordot_3_readvariableop_resource* 
_output_shapes
:
ДД*
dtype02
Tensordot_15/ReadVariableOpp
Tensordot_15/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_15/axesw
Tensordot_15/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_15/freea
Tensordot_15/ShapeShape	mul_4:z:0*
T0*
_output_shapes
:2
Tensordot_15/Shapez
Tensordot_15/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_15/GatherV2/axisр
Tensordot_15/GatherV2GatherV2Tensordot_15/Shape:output:0Tensordot_15/free:output:0#Tensordot_15/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_15/GatherV2~
Tensordot_15/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_15/GatherV2_1/axisц
Tensordot_15/GatherV2_1GatherV2Tensordot_15/Shape:output:0Tensordot_15/axes:output:0%Tensordot_15/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_15/GatherV2_1r
Tensordot_15/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_15/Const
Tensordot_15/ProdProdTensordot_15/GatherV2:output:0Tensordot_15/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_15/Prodv
Tensordot_15/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_15/Const_1
Tensordot_15/Prod_1Prod Tensordot_15/GatherV2_1:output:0Tensordot_15/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_15/Prod_1v
Tensordot_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_15/concat/axisП
Tensordot_15/concatConcatV2Tensordot_15/axes:output:0Tensordot_15/free:output:0!Tensordot_15/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_15/concat
Tensordot_15/stackPackTensordot_15/Prod_1:output:0Tensordot_15/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_15/stack
Tensordot_15/transpose	Transpose	mul_4:z:0Tensordot_15/concat:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_15/transposeЋ
Tensordot_15/ReshapeReshapeTensordot_15/transpose:y:0Tensordot_15/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_15/ReshapeЋ
Tensordot_15/MatMulMatMul#Tensordot_15/ReadVariableOp:value:0Tensordot_15/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_15/MatMulw
Tensordot_15/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_15/Const_2z
Tensordot_15/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_15/concat_1/axisЬ
Tensordot_15/concat_1ConcatV2Tensordot_15/Const_2:output:0Tensordot_15/GatherV2:output:0#Tensordot_15/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_15/concat_1
Tensordot_15ReshapeTensordot_15/MatMul:product:0Tensordot_15/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_15{
transpose_15/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_15/perm
transpose_15	TransposeTensordot_15:output:0transpose_15/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_15r
add_5AddV2transpose_14:y:0transpose_15:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
add_5
Tensordot_16/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Д@*
dtype02
Tensordot_16/ReadVariableOpp
Tensordot_16/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_16/axesw
Tensordot_16/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_16/freee
Tensordot_16/ShapeShapeImag:output:0*
T0*
_output_shapes
:2
Tensordot_16/Shapez
Tensordot_16/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_16/GatherV2/axisр
Tensordot_16/GatherV2GatherV2Tensordot_16/Shape:output:0Tensordot_16/free:output:0#Tensordot_16/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_16/GatherV2~
Tensordot_16/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_16/GatherV2_1/axisц
Tensordot_16/GatherV2_1GatherV2Tensordot_16/Shape:output:0Tensordot_16/axes:output:0%Tensordot_16/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_16/GatherV2_1r
Tensordot_16/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_16/Const
Tensordot_16/ProdProdTensordot_16/GatherV2:output:0Tensordot_16/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_16/Prodv
Tensordot_16/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_16/Const_1
Tensordot_16/Prod_1Prod Tensordot_16/GatherV2_1:output:0Tensordot_16/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_16/Prod_1v
Tensordot_16/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_16/concat/axisП
Tensordot_16/concatConcatV2Tensordot_16/axes:output:0Tensordot_16/free:output:0!Tensordot_16/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_16/concat
Tensordot_16/stackPackTensordot_16/Prod_1:output:0Tensordot_16/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_16/stack 
Tensordot_16/transpose	TransposeImag:output:0Tensordot_16/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
Tensordot_16/transposeЋ
Tensordot_16/ReshapeReshapeTensordot_16/transpose:y:0Tensordot_16/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_16/ReshapeЋ
Tensordot_16/MatMulMatMul#Tensordot_16/ReadVariableOp:value:0Tensordot_16/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_16/MatMulw
Tensordot_16/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_16/Const_2z
Tensordot_16/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_16/concat_1/axisЬ
Tensordot_16/concat_1ConcatV2Tensordot_16/Const_2:output:0Tensordot_16/GatherV2:output:0#Tensordot_16/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_16/concat_1
Tensordot_16ReshapeTensordot_16/MatMul:product:0Tensordot_16/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_16{
transpose_16/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_16/perm
transpose_16	TransposeTensordot_16:output:0transpose_16/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_16k
add_6AddV2	add_5:z:0transpose_16:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
add_6
Tensordot_17/ReadVariableOpReadVariableOp#tensordot_1_readvariableop_resource*
_output_shapes
:	Д@*
dtype02
Tensordot_17/ReadVariableOpp
Tensordot_17/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot_17/axesw
Tensordot_17/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot_17/freee
Tensordot_17/ShapeShapeReal:output:0*
T0*
_output_shapes
:2
Tensordot_17/Shapez
Tensordot_17/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_17/GatherV2/axisр
Tensordot_17/GatherV2GatherV2Tensordot_17/Shape:output:0Tensordot_17/free:output:0#Tensordot_17/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_17/GatherV2~
Tensordot_17/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_17/GatherV2_1/axisц
Tensordot_17/GatherV2_1GatherV2Tensordot_17/Shape:output:0Tensordot_17/axes:output:0%Tensordot_17/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot_17/GatherV2_1r
Tensordot_17/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_17/Const
Tensordot_17/ProdProdTensordot_17/GatherV2:output:0Tensordot_17/Const:output:0*
T0*
_output_shapes
: 2
Tensordot_17/Prodv
Tensordot_17/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot_17/Const_1
Tensordot_17/Prod_1Prod Tensordot_17/GatherV2_1:output:0Tensordot_17/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot_17/Prod_1v
Tensordot_17/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_17/concat/axisП
Tensordot_17/concatConcatV2Tensordot_17/axes:output:0Tensordot_17/free:output:0!Tensordot_17/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_17/concat
Tensordot_17/stackPackTensordot_17/Prod_1:output:0Tensordot_17/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot_17/stack 
Tensordot_17/transpose	TransposeReal:output:0Tensordot_17/concat:output:0*
T0*+
_output_shapes
:@џџџџџџџџџ2
Tensordot_17/transposeЋ
Tensordot_17/ReshapeReshapeTensordot_17/transpose:y:0Tensordot_17/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot_17/ReshapeЋ
Tensordot_17/MatMulMatMul#Tensordot_17/ReadVariableOp:value:0Tensordot_17/Reshape:output:0*
T0*(
_output_shapes
:Дџџџџџџџџџ2
Tensordot_17/MatMulw
Tensordot_17/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Д2
Tensordot_17/Const_2z
Tensordot_17/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot_17/concat_1/axisЬ
Tensordot_17/concat_1ConcatV2Tensordot_17/Const_2:output:0Tensordot_17/GatherV2:output:0#Tensordot_17/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot_17/concat_1
Tensordot_17ReshapeTensordot_17/MatMul:product:0Tensordot_17/concat_1:output:0*
T0*,
_output_shapes
:Дџџџџџџџџџ2
Tensordot_17{
transpose_17/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_17/perm
transpose_17	TransposeTensordot_17:output:0transpose_17/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
transpose_17k
add_7AddV2	add_6:z:0transpose_17:y:0*
T0*,
_output_shapes
:џџџџџџџџџД2
add_7l

norm_5/mulMul	add_7:z:0	add_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2

norm_5/mul
norm_5/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm_5/Sum/reduction_indices

norm_5/SumSumnorm_5/mul:z:0%norm_5/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџД*
	keep_dims(2

norm_5/Sumn
norm_5/SqrtSqrtnorm_5/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
norm_5/Sqrt
norm_5/SqueezeSqueezenorm_5/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџД*
squeeze_dims
2
norm_5/Squeezef
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_5/dim
ExpandDims_5
ExpandDimsnorm_5/Squeeze:output:0ExpandDims_5/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
ExpandDims_5y
Tile_5/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Tile_5/multiples
Tile_5TileExpandDims_5:output:0Tile_5/multiples:output:0*
T0*,
_output_shapes
:џџџџџџџџџД2
Tile_5
Maximum_5/ReadVariableOpReadVariableOp!maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02
Maximum_5/ReadVariableOp
	Maximum_5MaximumTile_5:output:0 Maximum_5/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	Maximum_5~
ReadVariableOp_5ReadVariableOp!maximum_4_readvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_5
	truediv_5RealDivReadVariableOp_5:value:0Maximum_5:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
	truediv_5W
sub_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_9/xm
sub_9Subsub_9/x:output:0truediv_5:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
sub_9b
mul_5Mul	sub_9:z:0	add_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџД2
mul_5\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2	mul_4:z:0	mul_5:z:0concat/axis:output:0*
N*
T0*,
_output_shapes
:џџџџџџџџџш2
concath
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџ@::::::::T P
+
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1
э
Х
$__inference_signature_wrapper_903618
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџш*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_9030132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџ@:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_1"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Д
serving_default 
?
input_14
serving_default_input_1:0џџџџџџџџџ@A
output_15
StatefulPartitionedCall:0џџџџџџџџџшtensorflow/serving/predict:Ё#
Т
Wt_r
Wt_i
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
-__call__
*.&call_and_return_all_conditional_losses
/_default_save_signature"Ж
_tf_keras_model{"class_name": "LISTA", "name": "lista_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "LISTA"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0001250000059371814, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-08, "amsgrad": false}}}}
:
ДД2Variable
:
ДД2Variable
:	Д@2Variable
:	Д@2Variable
5
0
1
2"
trackable_list_wrapper
: 2Variable
б

beta_1

beta_2
	decay
learning_rate
itermm m!m"m#m$m%v&v'v(v)v*v+v,"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
Ъ
	variables
non_trainable_variables

layers
metrics
	regularization_losses

trainable_variables
layer_metrics
layer_regularization_losses
-__call__
/_default_save_signature
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
,
0serving_default"
signature_map
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
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Л
	total
	count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
0
1"
trackable_list_wrapper
-
	variables"
_generic_user_object
!:
ДД2Adam/Variable/m
!:
ДД2Adam/Variable/m
 :	Д@2Adam/Variable/m
 :	Д@2Adam/Variable/m
: 2Adam/Variable/m
: 2Adam/Variable/m
: 2Adam/Variable/m
!:
ДД2Adam/Variable/v
!:
ДД2Adam/Variable/v
 :	Д@2Adam/Variable/v
 :	Д@2Adam/Variable/v
: 2Adam/Variable/v
: 2Adam/Variable/v
: 2Adam/Variable/v
ѕ2ђ
(__inference_lista_3_layer_call_fn_903589Х
В
FullArgSpec
args
jself
jY
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ **Ђ'
%"
input_1џџџџџџџџџ@
2
C__inference_lista_3_layer_call_and_return_conditional_losses_903569Х
В
FullArgSpec
args
jself
jY
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ **Ђ'
%"
input_1џџџџџџџџџ@
у2р
!__inference__wrapped_model_903013К
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ **Ђ'
%"
input_1џџџџџџџџџ@
3B1
$__inference_signature_wrapper_903618input_1
!__inference__wrapped_model_903013y4Ђ1
*Ђ'
%"
input_1џџџџџџџџџ@
Њ "8Њ5
3
output_1'$
output_1џџџџџџџџџшВ
C__inference_lista_3_layer_call_and_return_conditional_losses_903569k4Ђ1
*Ђ'
%"
input_1џџџџџџџџџ@
Њ "*Ђ'
 
0џџџџџџџџџш
 
(__inference_lista_3_layer_call_fn_903589^4Ђ1
*Ђ'
%"
input_1џџџџџџџџџ@
Њ "џџџџџџџџџш­
$__inference_signature_wrapper_903618?Ђ<
Ђ 
5Њ2
0
input_1%"
input_1џџџџџџџџџ@"8Њ5
3
output_1'$
output_1џџџџџџџџџш