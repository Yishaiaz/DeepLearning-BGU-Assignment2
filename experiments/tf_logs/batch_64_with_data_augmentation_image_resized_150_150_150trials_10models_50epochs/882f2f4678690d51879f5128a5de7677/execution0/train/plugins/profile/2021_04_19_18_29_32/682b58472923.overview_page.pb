?	??zO??h@??zO??h@!??zO??h@	?h?+A???h?+A??!?h?+A??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??zO??h@?????[@1??g?"U@A????y??I/??)@Y?_cD???*	??ʁm?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??I~?@!e+n???X@)̘?5>@1qʼM?V@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@h^????!9?e'@)????K???1y???Z?	@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@O?}????!?<|}pT@)S?Q?G??1?'pD?@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@K?????!?*r¥??)K?????1?*r¥??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchA?)V¬?!ӱh?$??)A?)V¬?1ӱh?$??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismv?d??!m6????)WAt???1?ƦF????:Preprocessing2F
Iterator::Model
?s34??!?&u????)9??U}?1W??.R??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?h?+A??I?n??_L@QjG??i?E@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????[@?????[@!?????[@      ??!       "	??g?"U@??g?"U@!??g?"U@*      ??!       2	????y??????y??!????y??:	/??)@/??)@!/??)@B      ??!       J	?_cD????_cD???!?_cD???R      ??!       Z	?_cD????_cD???!?_cD???b      ??!       JGPUY?h?+A??b q?n??_L@yjG??i?E@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterU??g})??!U??g})??0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?3??)??!?eƚJ)??0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput??k;ȣ?!h?9R???0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?&Q?Z???!F?Cf]??0"@
model/sequential/conv2d_1/Relu_FusedConv2D?k[9??!??Ȉ???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2Dy??@??!??]?????"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3ߐ?L?Y??!???L.??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3Q?.P?6??!?|???"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter9?_???!??Z????0"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?xyښ?!9?&0Z???0Q      Y@Y?rO#,F@a>?????K@q%?oK`?.@y????(?m?"?

both?Your program is POTENTIALLY input-bound because 55.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?15.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 