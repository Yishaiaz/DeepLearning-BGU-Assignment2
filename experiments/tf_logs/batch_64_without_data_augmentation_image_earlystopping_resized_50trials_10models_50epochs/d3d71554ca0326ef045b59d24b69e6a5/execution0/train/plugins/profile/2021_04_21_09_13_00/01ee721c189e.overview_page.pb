?	whX??oZ@whX??oZ@!whX??oZ@	?OP??????OP?????!?OP?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6whX??oZ@e73???0@1]p??U@A4?l\??IAaP??$??Y???j?=??*	?I?Ư@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2>?4a??@!?=??!?X@)Y??w @1,S=WHV@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@(??h????!?L???@)(??h????1?L???@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@m???L??!=QU?!@)Q?+?Ͼ?1??j«@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@2Xq??0??!?jR???@)??N??1??X?( @:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchND??~???!|j?Nl??)ND??~???1|j?Nl??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??J̳???!?y+?;???)?????U??1?Ƃ??;??:Preprocessing2F
Iterator::Model??N?`???!.???????)S?A?Ѫv?1Β(?fj??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?OP?????I?????1@Q?q3(ImT@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	e73???0@e73???0@!e73???0@      ??!       "	]p??U@]p??U@!]p??U@*      ??!       2	4?l\??4?l\??!4?l\??:	AaP??$??AaP??$??!AaP??$??B      ??!       J	???j?=?????j?=??!???j?=??R      ??!       Z	???j?=?????j?=??!???j?=??b      ??!       JGPUY?OP?????b q?????1@y?q3(ImT@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter??8????!??8????0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterz<Л????!??'j????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput?B??G??!??(?*??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput[`,??=??!4??i???0"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D$2?۫???!`ZR"???"@
model/sequential/conv2d_1/Relu_FusedConv2D?ڎJ????!?5?K1<??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??????!?F? ?Z??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3??j?ՠ?!N??0(u??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????GI??!???Wތ??0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter'7??Q-??!V?&?H^??0Q      Y@YZ??]q?@@a? 6QG?P@qG?#??@y]????js?"?	
both?Your program is POTENTIALLY input-bound because 16.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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