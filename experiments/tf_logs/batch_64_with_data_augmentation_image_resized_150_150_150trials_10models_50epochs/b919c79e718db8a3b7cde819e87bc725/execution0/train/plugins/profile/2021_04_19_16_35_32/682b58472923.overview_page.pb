?	9{g?պk@9{g?պk@!9{g?պk@	W??????W??????!W??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails69{g?պk@?
?rG`@1????%V@A)???^??Iv?TQ@Y???2??*	?O??n??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2????@!??;??X@)=?බ @1Ta?@yW@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@ ??????!?Td?]C@)@k~??E??1?????m@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@???A???!??[?	@)?|?b?:??1?{?n?p??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@àL??Ű?!D?I????)àL??Ű?1D?I????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???m??!?́V???)???m??1?́V???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismd??u??!O??N???)>???6??1?!?4?k??:Preprocessing2F
Iterator::ModelܷZ'.ǳ?!?Wq9???)X<?H??z?1?	??T7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9W??????IF?,??M@Q???Nl?C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?
?rG`@?
?rG`@!?
?rG`@      ??!       "	????%V@????%V@!????%V@*      ??!       2	)???^??)???^??!)???^??:	v?TQ@v?TQ@!v?TQ@B      ??!       J	???2?????2??!???2??R      ??!       Z	???2?????2??!???2??b      ??!       JGPUYW??????b qF?,??M@y???Nl?C@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter^=w?v??!^=w?v??0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter???1	??!??T
??0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput???????!d??/?|??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?>^d˂??!:?f?L???0"@
model/sequential/conv2d_1/Relu_FusedConv2DgT?????!?%1??I??"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D????d¢?!?Dpk???"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3Sd??uΠ?!/?O)߻??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3ٜ뗭???!?dM????"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterzL?#(??!?,݆K???0"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterOa:o%??!??Vbt???0Q      Y@YZ??]q?@@a? 6QG?P@q'n???0@y?˽b??r?"?

both?Your program is POTENTIALLY input-bound because 58.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 