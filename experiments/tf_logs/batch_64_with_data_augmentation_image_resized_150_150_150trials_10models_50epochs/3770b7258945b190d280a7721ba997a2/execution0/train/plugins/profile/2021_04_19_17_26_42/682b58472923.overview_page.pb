?	?d?VA'k@?d?VA'k@!?d?VA'k@	T;??$??T;??$??!T;??$??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?d?VA'k@????"`@1s?w??CU@A??%ǝұ?I?KK#@Yp?h?????*	6^?ILP?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??EB[n@!?	c5?X@)?ꐛ?&@1?o?I?V@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@yX?5?;??!?h!??^@)rM??΢??1?`(?z	@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@9
3???!s8%h?@)??bb?q??1p<s"?E@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?H??rڳ?!?h~O????)?H??rڳ?1?h~O????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?5?o????!?3b????)?5?o????1?3b????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????R???!W??ê??)?????Q??1c????:Preprocessing2F
Iterator::Modelx?ܙ	???!??}??r??)ʩ?ajK}?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9T;??$??Iږ??ON@Q?P?zؓC@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????"`@????"`@!????"`@      ??!       "	s?w??CU@s?w??CU@!s?w??CU@*      ??!       2	??%ǝұ???%ǝұ?!??%ǝұ?:	?KK#@?KK#@!?KK#@B      ??!       J	p?h?????p?h?????!p?h?????R      ??!       Z	p?h?????p?h?????!p?h?????b      ??!       JGPUYT;??$??b qږ??ON@y?P?zؓC@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterǶ??3	??!Ƕ??3	??0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterw????!MR_???0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput??.?W???!6)?T?x??0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput??.?W???!MJ2???0"@
model/sequential/conv2d_1/Relu_FusedConv2D?T_i????!??I?G??"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D????????!(!I?ğ??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3Ldx?'R??!?-X?	???"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3e"`?*??!?n\`???"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterӘ??;???!???????0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterӘ??;???!?Ј?#???0Q      Y@YZ??]q?@@a? 6QG?P@qH5?+3@yUm?Z?m?"?

both?Your program is POTENTIALLY input-bound because 59.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?19.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 