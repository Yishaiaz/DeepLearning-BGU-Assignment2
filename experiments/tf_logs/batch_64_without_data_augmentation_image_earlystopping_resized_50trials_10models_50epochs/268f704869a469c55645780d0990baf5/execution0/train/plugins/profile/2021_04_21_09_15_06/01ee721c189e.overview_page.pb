?	??V?[@??V?[@!??V?[@	:\W47/??:\W47/??!:\W47/??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??V?[@[?}?6@1?8~??U@A???9????I??Q??Z??YV?P??r??*	?ʡE6 ?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2w??o?I@!?.k??X@)?????t@1??t?W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???͎T??!}1?kv@)?jJ?G??1?=TGe@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@R??b??!g%?ȥ?@)B^&?ǯ?1?V?Om???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@b???X???!???A?M??)b???X???1???A?M??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?1!撪??!+?&6????)?1!撪??1+?&6????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?u?r???!G?{k???)\sG?˵??1:?ى?a??:Preprocessing2F
Iterator::Model????0???!U?M4eW??)v?!H{?1o???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9:\W47/??I?ZF?6@Q????SS@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	[?}?6@[?}?6@![?}?6@      ??!       "	?8~??U@?8~??U@!?8~??U@*      ??!       2	???9???????9????!???9????:	??Q??Z????Q??Z??!??Q??Z??B      ??!       J	V?P??r??V?P??r??!V?P??r??R      ??!       Z	V?P??r??V?P??r??!V?P??r??b      ??!       JGPUY:\W47/??b q?ZF?6@y????SS@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?^Wݼ???!?^Wݼ???0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?q?9۾??!@h?????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput[?[??8??!K?~+?'??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput???{?5??!?#?????0"@
model/sequential/conv2d_1/Relu_FusedConv2D<2 ౢ?!Ne????"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D*$?????!?ꦓ?5??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3?'3D??!?JX??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3??W?Ӡ?!.?D?r??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterw???:??!?z??2???0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter??2?*??!?>@??\??0Q      Y@YZ??]q?@@a? 6QG?P@q(ܻY"?'@yS1?*is?"?

both?Your program is POTENTIALLY input-bound because 20.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?11.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 