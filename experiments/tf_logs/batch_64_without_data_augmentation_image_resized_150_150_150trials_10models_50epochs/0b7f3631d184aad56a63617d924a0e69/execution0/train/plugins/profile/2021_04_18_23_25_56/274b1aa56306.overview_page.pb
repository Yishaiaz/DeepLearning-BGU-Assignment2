?	`?;?	i@`?;?	i@!`?;?	i@	???M<5?????M<5??!???M<5??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6`?;?	i@x?q?Z?Z@1???Dh{V@A?/EHݲ?I?? @Y?l??3H??*	_?I??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2???Bt8@!tP?Qr?X@)g??6@1y???SW@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@\??&??!??q#?@)j?????1??o?%? @:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?Ӂ??V??!?k?:!?@)??׻??1a??ԼF??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@i? ?w???!?B???C??)i? ?w???1?B???C??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch_?L?J??!p?Y?w???)_?L?J??1p?Y?w???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???????!Vo???m??)?R?!?u??1?C{??\??:Preprocessing2F
Iterator::Modelv??^
??!??k?kc??)tys?V{x?1287[?[??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???M<5??IդSK@Q;?$?dF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	x?q?Z?Z@x?q?Z?Z@!x?q?Z?Z@      ??!       "	???Dh{V@???Dh{V@!???Dh{V@*      ??!       2	?/EHݲ??/EHݲ?!?/EHݲ?:	?? @?? @!?? @B      ??!       J	?l??3H???l??3H??!?l??3H??R      ??!       Z	?l??3H???l??3H??!?l??3H??b      ??!       JGPUY???M<5??b qդSK@y;?$?dF@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter??d?O???!??d?O???0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ka?????!?
?r???0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput???8R??!?`4?1??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput)???Z7??!???????0"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D9?>ɓ???!?Fw???"@
model/sequential/conv2d_1/Relu_FusedConv2D/S??!???!??L?@??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3ωbsi??!dѸ)N??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3?r`?V??!?8?X??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???Lo???!KtI???0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterY??#s???!?h??0Q      Y@YZ??]q?@@a? 6QG?P@q?fPz?#@y??ڱ??k?"?	
both?Your program is POTENTIALLY input-bound because 53.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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