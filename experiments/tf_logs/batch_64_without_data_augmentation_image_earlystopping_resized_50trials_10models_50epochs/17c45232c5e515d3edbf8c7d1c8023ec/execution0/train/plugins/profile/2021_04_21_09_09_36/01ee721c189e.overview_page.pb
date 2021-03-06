?	Z????A[@Z????A[@!Z????A[@	1]????1]????!1]????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Z????A[@?K???T4@1?5?!?U@A?{???S??I??	j???Y?,?????*	?(\??X?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2whX???@!kܲ??X@){?V???@1?R??:?V@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?'?????!?Kt??? @)??%:?,??1?U?
@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@Y??????!?ΐy?@)Y??????1?ΐy?@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??????!??]?ֹ@)?|]??t??1???3?@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchx}??O9??!????8???)x}??O9??1????8???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism+?򑔬?!?z?f???)[??	m??1?K?t??:Preprocessing2F
Iterator::Model??.o??!D?H]???)??~???s?1????g߿?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no91]????Ih:?e>?3@Q??6??S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?K???T4@?K???T4@!?K???T4@      ??!       "	?5?!?U@?5?!?U@!?5?!?U@*      ??!       2	?{???S???{???S??!?{???S??:	??	j?????	j???!??	j???B      ??!       J	?,??????,?????!?,?????R      ??!       Z	?,??????,?????!?,?????b      ??!       JGPUY1]????b qh:?e>?3@y??6??S@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterʆ? ѽ??!ʆ? ѽ??0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter_єS}???!,):????0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?O?rE:??!vy??#??0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput???`?2??!???T@???0"@
model/sequential/conv2d_1/Relu_FusedConv2Dc???Y???!?Fi?????"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?
?:????!?Ǭ?0??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??C?)ڠ?!?fp?$L??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3%/N)?ՠ?!?,?0?f??"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterw0܏?0??!???T????0"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?J?*??!'N?lDV??0Q      Y@YZ??]q?@@a? 6QG?P@qHQ??j@y?V?9Nm?"?	
both?Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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