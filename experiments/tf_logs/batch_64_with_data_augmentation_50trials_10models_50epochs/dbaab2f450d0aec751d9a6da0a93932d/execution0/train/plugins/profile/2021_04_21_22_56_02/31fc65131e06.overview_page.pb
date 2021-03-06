?	?'???k@?'???k@!?'???k@	?a???%???a???%??!?a???%??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?'???k@?׻?@1:?۠?Bj@A??????p?Ii9?Cm@Y-$`tys??*	k?t?h?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?y???{(@!G????X@)?j?=&?'@19?s?|X@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@m??????!??I6?? @)?2???y??1??m???:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??a??4??!???S??@))^emS<??1??Wu̧??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@J?_????!F?ӧ????)J?_????1F?ӧ????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch=,Ԛ???!??oq????)=,Ԛ???1??oq????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??S?????!]?d????)p&?????1?K?>?A??:Preprocessing2F
Iterator::Model?p!??F??!?ܭw???)????{?1G??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?a???%??I`???p@QĘ?2hFX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?׻?@?׻?@!?׻?@      ??!       "	:?۠?Bj@:?۠?Bj@!:?۠?Bj@*      ??!       2	??????p???????p?!??????p?:	i9?Cm@i9?Cm@!i9?Cm@B      ??!       J	-$`tys??-$`tys??!-$`tys??R      ??!       Z	-$`tys??-$`tys??!-$`tys??b      ??!       JGPUY?a???%??b q`???p@yĘ?2hFX@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?B??ݭ?!?B??ݭ?0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterr?5g!ݭ?!??Asݽ?0"@
model/sequential/conv2d_1/Relu_FusedConv2D???bVk??!??f9?I??"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D????][??!????f???"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??A????!???<g???"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3i?߉?ܨ?!P??-??"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput??;?;??!??:?s???0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInputظ?K83??!N??ژ??0"-
IteratorGetNext/_1_Send?8x?7??!??<6Y\??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??K?z??!ꂁ? ??0Q      Y@Y?e?Gʪ=@a?&nM?Q@qG?e#?@y??+???W?"?	
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