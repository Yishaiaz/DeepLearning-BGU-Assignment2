?	?B?n6h@?B?n6h@!?B?n6h@	f;????f;????!f;????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?B?n6h@?p!??Y@1[???X#V@A7o??=??I??'?.???Y	???W??*	?????g?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2K?ó?@!Pޮm?X@)jm?k?@1.?.?H?V@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?R???!?????@)1
?Ƿw??1??'?%?@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@c	kc????!3???M@)?qp?????1H?????@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?Ss??P??!y
?? M??)?Ss??P??1y
?? M??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??5?K??!?*}P?U??)??5?K??1?*}P?U??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismmscz???!?)C???);Qi??1???n%??:Preprocessing2F
Iterator::Model*U??-???!??xH?$??)9?	?ʼu?1???,??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9f;????I?fk?J@Qmy???F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?p!??Y@?p!??Y@!?p!??Y@      ??!       "	[???X#V@[???X#V@![???X#V@*      ??!       2	7o??=??7o??=??!7o??=??:	??'?.?????'?.???!??'?.???B      ??!       J		???W??	???W??!	???W??R      ??!       Z		???W??	???W??!	???W??b      ??!       JGPUYf;????b q?fk?J@ymy???F@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter? ????!? ????0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFiltert?)????!?6??G??0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput<??
???!
WY$I???0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput)??4????!?O?
????0"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?bRȭ???!???&V??"@
model/sequential/conv2d_1/Relu_FusedConv2D??,?eآ?!{.w3???"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3???x???!1AE?????"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3?`?aH???!P-??????"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterΉו+??!?{xB???0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterO̬?]??!Xyf????0Q      Y@YZ??]q?@@a? 6QG?P@ql?׏8%@y??o!?r?"?

both?Your program is POTENTIALLY input-bound because 52.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?10.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 