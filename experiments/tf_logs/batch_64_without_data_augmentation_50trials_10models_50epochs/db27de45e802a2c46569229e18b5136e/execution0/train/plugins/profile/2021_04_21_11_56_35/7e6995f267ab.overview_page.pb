?	?UJ?tu@?UJ?tu@!?UJ?tu@      ??!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?UJ?tu@cD?в.	@14?Y?w?t@I?h?'?O@*	/?$?U?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?>V?۰%@!???:?X@)3m??J#%@1gy?;0X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?3ڪ$???!;??Q??@)W|C??u??1?Z+~?H??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@P?eo)??!?A?%C7??)i7????1n?.???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@7+1?J??!?n????)7+1?J??1?n????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????X???!l> ????)????X???1l> ????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??n?;2??!??$?~???)????=z??1??Q??I??:Preprocessing2F
Iterator::Modelr?)????!???????)????|?1E??WB???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?Y?lA/??Q?
L?B?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	cD?в.	@cD?в.	@!cD?в.	@      ??!       "	4?Y?w?t@4?Y?w?t@!4?Y?w?t@*      ??!       2      ??!       :	?h?'?O@?h?'?O@!?h?'?O@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Y?lA/??y?
L?B?X@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?^;?ү?!?^;?ү?0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?w6G$į?!?#??˿?0"@
model/sequential/conv2d_1/Relu_FusedConv2D?ܝI?I??!?l??3???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?????3??!s-k??"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInputrR?,??!??`y)???0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput?H|ҋ+??!?V???M??0"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3(??:^=??!???F???"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3?Z??F/??!:??/[??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???,ރ??!??5W????0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?UF??m??!??>nh???0Q      Y@Y?e?Gʪ=@a?&nM?Q@q!G&i_+@y?B?H?Q?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?13.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 