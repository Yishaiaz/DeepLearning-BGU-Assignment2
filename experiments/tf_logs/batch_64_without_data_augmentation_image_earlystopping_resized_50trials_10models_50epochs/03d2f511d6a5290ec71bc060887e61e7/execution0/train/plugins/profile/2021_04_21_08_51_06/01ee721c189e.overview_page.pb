?	[??Y~^@[??Y~^@![??Y~^@	ڍt?????ڍt?????!ڍt?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6[??Y~^@?#??tA@1C=}?9U@Ao??}U.??I?uii??Y?$>w?=??*	?z?gа@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?U???@!??lv??X@)??=x?@1m?iW?-W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@UQ??ڦ??!?75??@@)????%ƺ?1u?Zp@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@????C??!??h??	@)y??[Y???1XP(pC???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@
K<?lʭ?!BG??ɠ??)
K<?lʭ?1BG??ɠ??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchOʤ?6 ??!??TE???)Oʤ?6 ??1??TE???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?I?p??!{???xR??)??D.8???16?????:Preprocessing2F
Iterator::Model??\????!8??db???)8i?x?1??vpNo??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ڍt?????IX??*?t=@QNhy[gQ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?#??tA@?#??tA@!?#??tA@      ??!       "	C=}?9U@C=}?9U@!C=}?9U@*      ??!       2	o??}U.??o??}U.??!o??}U.??:	?uii???uii??!?uii??B      ??!       J	?$>w?=???$>w?=??!?$>w?=??R      ??!       Z	?$>w?=???$>w?=??!?$>w?=??b      ??!       JGPUYڍt?????b qX??*?t=@yNhy[gQ@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter`?x?????!`?x?????0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter_qͼ??!;??%e???0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInputI?)n$??!$Q??!??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput??ɸ4??!?Qj~????0"@
model/sequential/conv2d_1/Relu_FusedConv2Dt?sń??!E@?,R???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D????m??!?0?!?"??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3%t?S?*??!&?p?FH??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3!,#J?%??!?D?um??"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter??NSܙ?!ښPUc???0"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???1ƙ?!?????S??0Q      Y@YZ??]q?@@a? 6QG?P@q??X
??*@yCBһ?m?"?

both?Your program is POTENTIALLY input-bound because 27.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?13.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 