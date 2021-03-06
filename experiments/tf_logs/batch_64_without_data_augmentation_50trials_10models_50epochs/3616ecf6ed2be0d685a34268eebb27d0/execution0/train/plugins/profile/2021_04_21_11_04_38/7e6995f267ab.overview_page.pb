?	?ŦUfu@?ŦUfu@!?ŦUfu@	?lZJ?C???lZJ?C??!?lZJ?C??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?ŦUfu@?9@0G?@1F?W???t@AuV?1???I?W?????Y???{???*	a??"?_?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV25&?\R?)@!??ߒ??X@)a??L&)@1?X?h2X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@{??????!i}(?	@)???4????1?k:QT???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@am?????!?f??d???)am?????1?f??d???:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@@a??+??!4???$??)?K?K?1??1?????Z??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?? ?X4??!????f??)?? ?X4??1????f??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?)??ѫ?!T?????)????Fu??1	????t??:Preprocessing2F
Iterator::Model??Po??!?tc mH??)?뤾,?t?1???mj"??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?lZJ?C??I??¢??Qe];1?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?9@0G?@?9@0G?@!?9@0G?@      ??!       "	F?W???t@F?W???t@!F?W???t@*      ??!       2	uV?1???uV?1???!uV?1???:	?W??????W?????!?W?????B      ??!       J	???{??????{???!???{???R      ??!       Z	???{??????{???!???{???b      ??!       JGPUY?lZJ?C??b q??¢??ye];1?X@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter	۩0?r??!	۩0?r??0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??~yd??!?1Uuk??0"@
model/sequential/conv2d_1/Relu_FusedConv2Dާ???!?P?????"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?yytd??!Ban?	???"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput??Dt??!8??o???0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput=??$????!?dO4???0"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3 l՘??!`j???"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3`V?ɸ???!,=? ??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter|????D??!|sݜ?G??0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterl????;??!ʰ?to??0Q      Y@Y?e?Gʪ=@a?&nM?Q@q?!?s?q@y.:?bX?P?"?	
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