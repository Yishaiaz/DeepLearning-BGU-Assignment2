?	t??%`j@t??%`j@!t??%`j@	?B|j?????B|j????!?B|j????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6t??%`j@??A?^@1????P?U@A?#ӡ????I???o_??Y?̰Q?o??*	????+B?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2<?8b-^@!{C????X@)X?vMH{@1
??9?mW@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??E??\??!G??+?@).W?6ɏ??1?(??#??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?0&????!?y??b@)?}r 
??1??k???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???x???!?e&9v???)???x???1?e&9v???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@ȳ˷>??!gDw?4???)ȳ˷>??1gDw?4???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?\?????!??r9&P??)L6l?ۗ?1?j1?>??:Preprocessing2F
Iterator::Model?xy:W???!e!??ß??)MI???*}?1???|??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?B|j????IY???~?L@Q!w????D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??A?^@??A?^@!??A?^@      ??!       "	????P?U@????P?U@!????P?U@*      ??!       2	?#ӡ?????#ӡ????!?#ӡ????:	???o_?????o_??!???o_??B      ??!       J	?̰Q?o???̰Q?o??!?̰Q?o??R      ??!       Z	?̰Q?o???̰Q?o??!?̰Q?o??b      ??!       JGPUY?B|j????b qY???~?L@y!w????D@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5?ۦ_*??!5?ۦ_*??0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?Y?:)??!??>?)??0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput!??????!?=?<@???0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput,??ß??!2???8??0"@
model/sequential/conv2d_1/Relu_FusedConv2D?K??&??!??w??t??"B
 model/sequential/conv2d_1/Relu_1_FusedConv2DIo|?b֢?!?W??????"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3͵܇Ơ?!Knh?Z???"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3J?厌??!tQ?????"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?>(l??!?¼?W???0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?>׮i??!?|3?????0Q      Y@YZ??]q?@@a? 6QG?P@q????n0@y??z?ךl?"?

both?Your program is POTENTIALLY input-bound because 56.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 