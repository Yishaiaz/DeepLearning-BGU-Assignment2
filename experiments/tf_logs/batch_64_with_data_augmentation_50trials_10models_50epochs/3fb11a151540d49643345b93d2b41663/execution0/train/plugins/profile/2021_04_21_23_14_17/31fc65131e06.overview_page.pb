?	?B?5v3f@?B?5v3f@!?B?5v3f@	??u??????u????!??u????"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?B?5v3f@?0Bx??@1E?4~?Xe@I?@J????Y?K?^I??*	?Mb(??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?sI?%@!???`??X@)?E`?o?$@1+???TX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?[Y?????!Y5?N@)?Y,E????1??C?s??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@}]??t??!???????)?G??Q??1?????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@\sG?˵??!?Y??????)\sG?˵??1?Y??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Ue????!(?!fF??)? {???1OV?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchÚʢ????!j??!??)Úʢ????1j??!??:Preprocessing2F
Iterator::Model???????!?
??'??)_~?Ɍ?u?1??O54x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??u????I ??X?@Q?#???	X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?0Bx??@?0Bx??@!?0Bx??@      ??!       "	E?4~?Xe@E?4~?Xe@!E?4~?Xe@*      ??!       2      ??!       :	?@J?????@J????!?@J????B      ??!       J	?K?^I???K?^I??!?K?^I??R      ??!       Z	?K?^I???K?^I??!?K?^I??b      ??!       JGPUY??u????b q ??X?@y?#???	X@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?{?4???!?{?4???0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter??_????!L???t???0"@
model/sequential/conv2d_1/Relu_FusedConv2D??F?W???!???j????"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D????C˯?!X?#?-R??"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput???bI??!K~u?v???0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput??M?????!?1?R??0"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?#?C????!???????0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?t??????!??j???0"o
Cgradient_tape/model/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?#???/??!??h?{??0"q
Egradient_tape/model/sequential/conv2d_2/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter???)??!h?????0Q      Y@Ye??/vA@aM*}?DP@q??-Ϳn@y?-?B??c?"?	
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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