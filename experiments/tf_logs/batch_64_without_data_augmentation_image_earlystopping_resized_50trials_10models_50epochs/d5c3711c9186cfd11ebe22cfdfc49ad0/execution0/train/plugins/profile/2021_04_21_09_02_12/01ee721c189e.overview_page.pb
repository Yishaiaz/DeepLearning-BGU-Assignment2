?	?}9?`@?}9?`@!?}9?`@	mT?9N???mT?9N???!mT?9N???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?}9?`@-"???C@1?+=)?U@AK;5???I?d(??Yo?j{??*	u???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?]K?=@!t?Ԁ?X@)Z??m?S	@1?<>??V@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@&7??5???!?x?}B?@)?
E????1xzW?@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@ ??X???!֚?-/@)?Go??ܲ?14?>? @:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?4?BX??!????i??)?4?BX??1????i??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?C?M??!?wϧ???)?C?M??1?wϧ???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?-?\??!)?ߍ?:??)AEկt>??1JV??6l??:Preprocessing2F
Iterator::Model??t&??!??J?????)G??ǁw?1??Wc?(??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 30.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9mT?9N???I?p??@@Q??{ ?P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-"???C@-"???C@!-"???C@      ??!       "	?+=)?U@?+=)?U@!?+=)?U@*      ??!       2	K;5???K;5???!K;5???:	?d(???d(??!?d(??B      ??!       J	o?j{??o?j{??!o?j{??R      ??!       Z	o?j{??o?j{??!o?j{??b      ??!       JGPUYmT?9N???b q?p??@@y??{ ?P@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???^????!???^????0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterMuO?5???!??r????0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?/?C??!e@?????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput???FK???!?QY???0"@
model/sequential/conv2d_1/Relu_FusedConv2D? r?'???!??'?&???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?I?א???!0yN??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3QY???)??!ZJ?p?s??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3L2?T???!?0N?????"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?v?8K???!Dl7u???0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter9;~??!?F?eb??0Q      Y@YZ??]q?@@a? 6QG?P@q?ףxZp(@y????>m?"?

both?Your program is POTENTIALLY input-bound because 30.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?12.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 