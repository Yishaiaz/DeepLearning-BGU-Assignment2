?	????e@????e@!????e@	㺅6???㺅6???!㺅6???"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-????e@?#0??@1f/[5e@I?)[$???Y͕A?????*	?x?&Q]?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV21{?v??-@!*?ެ?X@)???@?u-@1	???J~X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@X?vMH??!???@?X??)??`?H??12??????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@????Bt??!S??,C\??)????Bt??1S??,C\??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?z0)>??!7?˱M??)i7????1E7X???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch1~??7??!F7
v??)1~??7??1F7
v??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??%ǝұ?!??ע??)5&?\R???1?=?+???:Preprocessing2F
Iterator::Model`̖??p??!@?j??)??)?W??y?1Z?'?J???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9㺅6???I ??@Q߁?,)X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?#0??@?#0??@!?#0??@      ??!       "	f/[5e@f/[5e@!f/[5e@*      ??!       2      ??!       :	?)[$????)[$???!?)[$???B      ??!       J	͕A?????͕A?????!͕A?????R      ??!       Z	͕A?????͕A?????!͕A?????b      ??!       JGPUY㺅6???b q ??@y߁?,)X@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltercx?~ղ?!cx?~ղ?0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterՠ ?]???!??L,????0"@
model/sequential/conv2d_1/Relu_FusedConv2D0??<??!???z????"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D,p?ê???!?ȕ?U??"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?S?ד"??!?%????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput??7_٫?!T?\.U??0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?B???Π?!??QP o??0"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????/̠?!C??L????0"q
Egradient_tape/model/sequential/conv2d_2/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter	???|_??!d?~N??0"o
Cgradient_tape/model/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???iT??!v$T\?	??0Q      Y@Ye??/vA@aM*}?DP@q?_??2e@y?=+???c?"?	
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 