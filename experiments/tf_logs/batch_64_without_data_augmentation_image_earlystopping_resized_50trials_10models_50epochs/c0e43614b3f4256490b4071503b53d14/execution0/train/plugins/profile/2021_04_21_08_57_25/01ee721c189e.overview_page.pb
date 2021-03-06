?	????zD_@????zD_@!????zD_@	ܦf?_??ܦf?_??!ܦf?_??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6????zD_@b???C@1MM?7?	U@A?T?:ƭ?I]??????Y?:?/K???*	?t??I?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2^??Y@!C???vX@)????y?@1????W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?G7¢"??!?:A???@)ʤ?6 ??1?0?I2n@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@0???"??!?D??P?	@)?)?n???1??q=??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch
K<?lʭ?!????Z??)
K<?lʭ?1????Z??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?,|}?K??!o?0???)?,|}?K??1o?0???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismjP4`???!????????)??X?????1?v'?n!??:Preprocessing2F
Iterator::Model?X???F??!?W?ȅ&@)n??KX{?1?~/??
??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 30.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ܦf?_??I?? ?M@@Q>?6?P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	b???C@b???C@!b???C@      ??!       "	MM?7?	U@MM?7?	U@!MM?7?	U@*      ??!       2	?T?:ƭ??T?:ƭ?!?T?:ƭ?:	]??????]??????!]??????B      ??!       J	?:?/K????:?/K???!?:?/K???R      ??!       Z	?:?/K????:?/K???!?:?/K???b      ??!       JGPUYܦf?_??b q?? ?M@@y>?6?P@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?X?b????!?X?b????0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterAc#????!5?BN???0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput?O?ZS??!?U??U??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInputV'?A?1??! d?%????0"@
model/sequential/conv2d_1/Relu_FusedConv2De̥f????!?e	??"B
 model/sequential/conv2d_1/Relu_1_FusedConv2DH.??}??!???g?c??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??HȡF??!+??????"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3?6?L?A??!JL?̴??"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter\????!??}?????0"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterT?+~?
??!?Eo?C{??0Q      Y@YZ??]q?@@a? 6QG?P@q? 5d??&@y??????m?"?

both?Your program is POTENTIALLY input-bound because 30.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?11.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 