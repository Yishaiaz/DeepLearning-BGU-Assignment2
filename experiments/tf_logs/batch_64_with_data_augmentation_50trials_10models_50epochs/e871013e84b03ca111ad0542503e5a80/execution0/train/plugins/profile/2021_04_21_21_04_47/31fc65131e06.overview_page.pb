?	q?J[?k@q?J[?k@!q?J[?k@	|?䋿??|?䋿??!|?䋿??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-q?J[?k@?&?f?@1N&nDIj@I臭??\@Y8-x?W??*	R??;??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??"?'@!?`s?l?X@)Ǽ?8d?&@1?*???DX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???z???!dv?z????)V?)??%??1?I;)????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??????!?)??????)??????1?)??????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?DeÚ???!???? ??)ū?m???1LH?g????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@Ή=????!??$?2??)Ή=????1??$?2??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismc??????!???????)}>ʈ@??1U?Bۢ??:Preprocessing2F
Iterator::Model???{*???!A?OƷ???)E|V|?13?+T`??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9|?䋿??I`??x?@Q?z?y6:X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?&?f?@?&?f?@!?&?f?@      ??!       "	N&nDIj@N&nDIj@!N&nDIj@*      ??!       2      ??!       :	臭??\@臭??\@!臭??\@B      ??!       J	8-x?W??8-x?W??!8-x?W??R      ??!       Z	8-x?W??8-x?W??!8-x?W??b      ??!       JGPUY|?䋿??b q`??x?@y?z?y6:X@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter????ޭ?!????ޭ?0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterwp?50ڭ?!__ܽ?0"@
model/sequential/conv2d_1/Relu_FusedConv2Ds??UNf??!\?ŠG??"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D??~oT??!?얤????"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3?M?9?%??!  ?Y???"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??>~̨?!?????"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInputm??Q?$??!?0@?A???0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInputjs?'? ??!O4)`???0"-
IteratorGetNext/_1_Sends?Y_Z??!FJ[??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?L;????!?mc???0Q      Y@Y?e?Gʪ=@a?&nM?Q@q?Ŗe?+@y???7?W?"?	
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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