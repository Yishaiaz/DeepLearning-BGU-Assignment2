?	ŪA?[?e@ŪA?[?e@!ŪA?[?e@	Z??T????Z??T????!Z??T????"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ŪA?[?e@A?} R?
@1܀???d@Io?$????Y$)?ahu??*	??Q????@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??????(@!?k?T?X@)Zf?? (@1q?HvX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@@i?QH2??!?8?@)?pvk???1???(????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismX??????!??>????)<Mf?????1͸]z???:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?a?7?W??!F!?H???)???,z??1?!7?^??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?????5??!veZJ??)?????5??1veZJ??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchs??c?Ȱ?!??tj,???)s??c?Ȱ?1??tj,???:Preprocessing2F
Iterator::ModeloF?W????!?e|Ϊ??)????x!}?1I??g???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Z??T????I ?B??@Q?	q(2*X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	A?} R?
@A?} R?
@!A?} R?
@      ??!       "	܀???d@܀???d@!܀???d@*      ??!       2      ??!       :	o?$????o?$????!o?$????B      ??!       J	$)?ahu??$)?ahu??!$)?ahu??R      ??!       Z	$)?ahu??$)?ahu??!$)?ahu??b      ??!       JGPUYZ??T????b q ?B??@y?	q(2*X@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?y"????!?y"????0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter???? ???!??_???0"@
model/sequential/conv2d_1/Relu_FusedConv2D??420??!?u?%x???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?/??i??!??t????"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput?z|?>??!v?V???0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput$??V_5??!L?A????0"-
IteratorGetNext/_1_Send???? p??!??V????"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?\~???!????:???0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?AK?????!?Md????0"q
Egradient_tape/model/sequential/conv2d_2/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterQ$?x???!????gc??0Q      Y@Ye??/vA@aM*}?DP@q\h?'7@y?????]?"?	
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