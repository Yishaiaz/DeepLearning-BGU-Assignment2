?	?U?@X?f@?U?@X?f@!?U?@X?f@	/??<e???/??<e???!/??<e???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?U?@X?f@???u??V@1???V@A???W???I??ԱJI@YyW=`2??*	?????Ȯ@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2????k?@!>?Mp?wX@)
j??m@1,/p?ӋV@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?+???d??!)A???@)|?/????1S??0-@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??Ϸ??!?z?@)?Rz????1???(Q@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@V?P?????!?|????)V?P?????1?|????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch2??z?p??!??͆M??)2??z?p??1??͆M??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??4?(??!^?E?????)W?9?m?1ĕ??????:Preprocessing2F
Iterator::Model4w??o??!+?M?1 @)h?o}Xot?1????4??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no90??<e???In??x??I@Q??'"?;H@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???u??V@???u??V@!???u??V@      ??!       "	???V@???V@!???V@*      ??!       2	???W??????W???!???W???:	??ԱJI@??ԱJI@!??ԱJI@B      ??!       J	yW=`2??yW=`2??!yW=`2??R      ??!       Z	yW=`2??yW=`2??!yW=`2??b      ??!       JGPUY0??<e???b qn??x??I@y??'"?;H@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterK?!????!K?!????0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterV{	????!?P?????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput?ԥ?|??!&?`?WN??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Z۰}d??!w?|@????0"@
model/sequential/conv2d_1/Relu_FusedConv2D??h+@???!jW?E???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?u??͡??!(???k??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??0?????!?%???~??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3??[?????!C?2?9???"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter????o??!~?? ????0"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterA?GJ?l??!??2??0Q      Y@Y?rO#,F@a>?????K@qBЬ?%@y??Wb's?"?

both?Your program is POTENTIALLY input-bound because 50.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?10.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 