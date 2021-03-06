?	??hq?f@??hq?f@!??hq?f@	Wv?4???Wv?4???!Wv?4???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??hq?f@D?R?V@1ӆ???wU@A??? ?r??I?nض(???Y??v????*	????&Գ@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2P9&???@!?ۓ???X@)p	????@1??.c2^W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??b?dU??!??R?T@)?8?Z????1????W@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@L??1%??!X,ؽ??@) C?*q??1y_'S?y??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??4*p???!6??(>H??)??4*p???16??(>H??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchb0?̕??!v?m???)b0?̕??1v?m???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismݔ?Z	ݱ?!t??????)???`?H??1???aP???:Preprocessing2F
Iterator::Model?q ????!t	???)?? Z+z?1Hp
H??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Xv?4???I?M?Y^I@QGھӟGH@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	D?R?V@D?R?V@!D?R?V@      ??!       "	ӆ???wU@ӆ???wU@!ӆ???wU@*      ??!       2	??? ?r????? ?r??!??? ?r??:	?nض(????nض(???!?nض(???B      ??!       J	??v??????v????!??v????R      ??!       Z	??v??????v????!??v????b      ??!       JGPUYXv?4???b q?M?Y^I@yGھӟGH@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter[?p?????![?p?????0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter???????!?U??????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput8ٲp^???!!??yER??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput???^???!5g;????0"@
model/sequential/conv2d_1/Relu_FusedConv2D??`K/Ѣ?!4?$???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?x@?????!P#[??p??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3>???!?&??q???"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3lq??? ??!?Fِ???"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?????i??!QI5????0"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter@?q?/`??!ؐq??0Q      Y@Y?rO#,F@a>?????K@q??q+?K%@y?l?CmNm?"?

both?Your program is POTENTIALLY input-bound because 49.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?10.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 