?	?{?5Zif@?{?5Zif@!?{?5Zif@	u߮????u߮????!u߮????"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?{?5Zif@?)???@1/???u?e@I?Cl?p???Y5E?ӻx??*	|?5^zj?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV28???'@!O?Xz?X@)?ǚ?A&'@18???"X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?Χ?U??!?<?Us3@)?,??2??1??????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@???????!?y0?cX??)?30??&??1?;Sfk??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@kdWZF???!soܷ0??)kdWZF???1soܷ0??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?7? ?ث?!?2????)?7? ?ث?1?2????:Preprocessing2F
Iterator::Model?̒ 5???!eqX?????)?.?.??1???&=??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?GW????!-??]????)O?\?	??1??Zf????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9u߮????I ????@Q??$-,X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)???@?)???@!?)???@      ??!       "	/???u?e@/???u?e@!/???u?e@*      ??!       2      ??!       :	?Cl?p????Cl?p???!?Cl?p???B      ??!       J	5E?ӻx??5E?ӻx??!5E?ӻx??R      ??!       Z	5E?ӻx??5E?ӻx??!5E?ӻx??b      ??!       JGPUYu߮????b q ????@y??$-,X@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?,?.?u??!?,?.?u??0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter|;.}?r??!4??:t??0"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D`?Zņc??!??L?M??"@
model/sequential/conv2d_1/Relu_FusedConv2Dp"?]>??!$ݞ?Y??"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInputiSӖyګ?!?Gy????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInputq?=d?٫?!?? ????0"-
IteratorGetNext/_1_Send^??????!+???t5??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter/U??ba??!?(U?a??0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter^?JʬR??!}???????0"o
Cgradient_tape/model/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterRၻ??!?MW\?#??0Q      Y@Ye??/vA@aM*}?DP@q^?`?$	@y]??h]?"?	
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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