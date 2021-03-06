?	Y??+?Xl@Y??+?Xl@!Y??+?Xl@	?h?+H???h?+H??!?h?+H??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Y??+?Xl@?R???H`@1??&7?W@A?4?($???I/??|????Ys۾G?u??*	??? ?.?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2ǀ????9@!??.?X@)?u?|π9@1aM???X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???????!c?;???)???4)??1?V?k?`??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@T?^P??!C??1/??)??l????1?8??*q??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch:?,B???!?xt?0??):?,B???1?xt?0??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?K?^I??!L??|??)?K?^I??1L??|??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?$xC??!p? w?x??)??Z
H???1?w2?|??:Preprocessing2F
Iterator::Model?>?Q?y??!PD;?????)??q?@Hv?1???7???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?h?+H??IV?? ??L@Q??I???D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?R???H`@?R???H`@!?R???H`@      ??!       "	??&7?W@??&7?W@!??&7?W@*      ??!       2	?4?($????4?($???!?4?($???:	/??|????/??|????!/??|????B      ??!       J	s۾G?u??s۾G?u??!s۾G?u??R      ??!       Z	s۾G?u??s۾G?u??!s۾G?u??b      ??!       JGPUY?h?+H??b qV?? ??L@y??I???D@?"-
IteratorGetNext/_1_Send?C??Zn??!?C??Zn??"-
IteratorGetNext/_3_Sendg2??
???!?????"N
#model_1/model/conv_dw_3/depthwise_1DepthwiseConv2dNative??a????!Kp??????"L
!model_1/model/conv_dw_3/depthwiseDepthwiseConv2dNative?D?q??!\??,????"L
!model_1/model/conv_dw_1/depthwiseDepthwiseConv2dNativeDk??u???!???>??"N
#model_1/model/conv_dw_1/depthwise_1DepthwiseConv2dNative??G???!<??????"7
model_1/model/conv_pad_2/Pad_1Pad??_?vߎ?!???????"5
model_1/model/conv_pad_2/PadPad_?~G܎?!???????"N
#model_1/model/conv_dw_2/depthwise_1DepthwiseConv2dNativeѪq?l??!\(?/:???"L
!model_1/model/conv_dw_2/depthwiseDepthwiseConv2dNative]?D6?J??!oMd9????Q      Y@Y?M?_{D@a>??b??M@q%$f?K?@y9jĒQrp?"?	
both?Your program is POTENTIALLY input-bound because 57.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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