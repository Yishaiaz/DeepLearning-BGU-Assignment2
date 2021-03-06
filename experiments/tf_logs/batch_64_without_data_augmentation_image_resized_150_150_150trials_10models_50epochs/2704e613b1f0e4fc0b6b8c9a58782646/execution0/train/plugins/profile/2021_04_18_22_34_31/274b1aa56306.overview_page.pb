?	̴?+??h@̴?+??h@!̴?+??h@	h2?aO??h2?aO??!h2?aO??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6̴?+??h@y??Z@1??Q?y?U@A?	j?֭?I??}q?J@Y???qť?*	??x?f?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??;??8@!s?2ǡ?X@)`;?O`@1?w#?V@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@T8?T???!??????@)??????1n?D@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@1?t?????!?_
P?|@)Ḍ?h??1?և?????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@ù???!????@??)ù???1????@??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch\?J???!????-???)\?J???1????-???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??)x
??!?d?>/???)eU???*??1B薯??:Preprocessing2F
Iterator::Modelc???&???!$cX3???)ŭ???w?1q?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9g2?aO??I?X?=?%L@Q-hA?E@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	y??Z@y??Z@!y??Z@      ??!       "	??Q?y?U@??Q?y?U@!??Q?y?U@*      ??!       2	?	j?֭??	j?֭?!?	j?֭?:	??}q?J@??}q?J@!??}q?J@B      ??!       J	???qť????qť?!???qť?R      ??!       Z	???qť????qť?!???qť?b      ??!       JGPUYg2?aO??b q?X?=?%L@y-hA?E@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter܃i?????!܃i?????0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter??9r????!H?Q????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput????d??!???]K??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput???3?0??!?ŁOn???0"B
 model/sequential/conv2d_1/Relu_1_FusedConv2Df????Ǣ?!??Dm
??"@
model/sequential/conv2d_1/Relu_FusedConv2D?9,?????!??Ż?^??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3????y8??!"???̅??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3Z?嬙Ԡ?!-??!`???"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter(G?0?^??!??Q?&???0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter???k?X??!???-?u??0Q      Y@YZ??]q?@@a? 6QG?P@qR 	\|;@y?^????s?"?

both?Your program is POTENTIALLY input-bound because 54.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?27.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 