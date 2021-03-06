?	??p9?g@??p9?g@!??p9?g@	lA??AV??lA??AV??!lA??AV??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??p9?g@ԛQ?UIY@1?\??$?U@AO?C?ͩ??I7?Nx	.@Y9}=_?\??*	?x?&1??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2~?N?Z@!???ǡX@)t??YE@1???[?W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@E?>?'I??!{9#_?V@)'??rJ@??1?\???+??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@c)??R??!yD?q??@)??w??1??1XHD?o8??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?m4??@??!?@L?????)?m4??@??1?@L?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?;p????!6w??`???)?;p????16w??`???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?N??唰?!s[???m??)?v?k??1]?????:Preprocessing2F
Iterator::ModelIe?9:??!Z_?????)?j?=&Rz?19 ????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9lA??AV??I??f_?EK@Q2??^?F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ԛQ?UIY@ԛQ?UIY@!ԛQ?UIY@      ??!       "	?\??$?U@?\??$?U@!?\??$?U@*      ??!       2	O?C?ͩ??O?C?ͩ??!O?C?ͩ??:	7?Nx	.@7?Nx	.@!7?Nx	.@B      ??!       J	9}=_?\??9}=_?\??!9}=_?\??R      ??!       Z	9}=_?\??9}=_?\??!9}=_?\??b      ??!       JGPUYlA??AV??b q??f_?EK@y2??^?F@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter@ z?????!@ z?????0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter? o)????!v?t?????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput	??????!???9??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?I??9??!?Z????0"@
model/sequential/conv2d_1/Relu_FusedConv2DK? 6????!?o??i???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D???9???!A-?7?J??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3m??{Ӡ?!>?d??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??S???!c??iz??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter`?Lq
B??!??v????0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter[??{2??!??bf?`??0Q      Y@YZ??]q?@@a? 6QG?P@qY??h?0@y?l?-m?"?

both?Your program is POTENTIALLY input-bound because 53.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 