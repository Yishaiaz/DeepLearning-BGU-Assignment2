?	B!??t@B!??t@!B!??t@	?p?%?"???p?%?"??!?p?%?"??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-B!??t@!??q4@1O?6???t@I?Z(?|@Y?N#-????*	?"???_?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?4ӽN?*@!gv)1??X@)?i?{??)@1??6X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@Eg?E(???!:Ob߹@)??=Զ??1?O?<???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@1`?U,~??!4??N1??)1`?U,~??14??N1??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?7/N|???!E$ևZ??)?????n??1#?x??Q??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch9?Վ???!?6?z????)9?Վ???1?6?z????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism/M??.??!P?ֈ??)?o?DIH??1??HW???:Preprocessing2F
Iterator::ModelY?n?ͷ?!<?Dk???)?27߈?y?1???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?p?%?"??I?J??????Q???T??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!??q4@!??q4@!!??q4@      ??!       "	O?6???t@O?6???t@!O?6???t@*      ??!       2      ??!       :	?Z(?|@?Z(?|@!?Z(?|@B      ??!       J	?N#-?????N#-????!?N#-????R      ??!       Z	?N#-?????N#-????!?N#-????b      ??!       JGPUY?p?%?"??b q?J??????y???T??X@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter!_<^??!!_<^??0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter2??Y???!?-:??0"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D????=v??!?h<??!??"@
model/sequential/conv2d_1/Relu_FusedConv2D>?bEt??!p	???>??"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput쇙C'Y??!??=?v???0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?
? nT??!?Xmu??0"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??B?7j??!???^K??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3?ކy?\??!?ұ?ݍ??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??ʞ{???!g0?A????0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterT6??4???!2???????0Q      Y@Y?e?Gʪ=@a?&nM?Q@qO?#_?@y??}>?P?"?	
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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