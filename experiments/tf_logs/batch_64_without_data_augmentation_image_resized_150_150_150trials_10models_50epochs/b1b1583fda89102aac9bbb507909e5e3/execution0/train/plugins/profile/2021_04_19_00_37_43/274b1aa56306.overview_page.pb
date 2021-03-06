?	??6S?i@??6S?i@!??6S?i@	?(+j????(+j???!?(+j???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??6S?i@Z????Z@1?q?
aV@A,*?t????I4????? @Y?c?????*	?O??n?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2"7???@!?|@6?X@)????@?@16?bk?W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?X5s???!~n?Q?X@)???Im??1?qT??@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@b?k_@??!?-?Nd?
@)?|??z???1mF??<???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@???&????!e??"??)???&????1e??"??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?ګ????!?{Ho??)?ګ????1?{Ho??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismV??y???!????3??)ZՒ?r0??1*tw?????:Preprocessing2F
Iterator::Model???
~??!????o2??)?&??d?v?1?Q?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?(+j???IL{?i?oK@Q?2???TF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z????Z@Z????Z@!Z????Z@      ??!       "	?q?
aV@?q?
aV@!?q?
aV@*      ??!       2	,*?t????,*?t????!,*?t????:	4????? @4????? @!4????? @B      ??!       J	?c??????c?????!?c?????R      ??!       Z	?c??????c?????!?c?????b      ??!       JGPUY?(+j???b qL{?i?oK@y?2???TF@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterq??????!q??????0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?K?{P???!(??.????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput?hƆ?w??!F~ş|K??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput\˨??\??!???z???0"@
model/sequential/conv2d_1/Relu_FusedConv2D???vŢ?!Hv?Z???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?UTs????!'??e??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3?a[?X??!v$?Ծp??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3?a[?X??!?G??{??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??g`????!(??
???0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter??fo?כ?!O~??{??0Q      Y@YZ??]q?@@a? 6QG?P@q?ӳ??1@y|F??l?"?

both?Your program is POTENTIALLY input-bound because 53.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?18.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 