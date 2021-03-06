?	????0m@????0m@!????0m@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-????0m@???^b@1!yv?U@A??E	???I?????? @*	??? `??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2H?3?9Y5@!e?d?X@)?k???(5@1b???˲X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@H???=??!2?#nL??)0??mP???1???З???:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@`#I????!&U????)n?8)?{??1?68????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@QhY?????!h???????)QhY?????1h???????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchy???h??!~?:A~??)y???h??1~?:A~??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?uS?k%??!?4?"????)e?,?i???1???f??:Preprocessing2F
Iterator::Model?tx㧱?!y??=????)??O?s'x?1k???1??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI"MЯ??O@Q޲/PmB@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???^b@???^b@!???^b@      ??!       "	!yv?U@!yv?U@!!yv?U@*      ??!       2	??E	?????E	???!??E	???:	?????? @?????? @!?????? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q"MЯ??O@y޲/PmB@?"-
IteratorGetNext/_2_Recv?G'?@??!?G'?@??"-
IteratorGetNext/_4_Recv?ǈ??;??!xXX>??"N
#model_1/model/conv_dw_3/depthwise_1DepthwiseConv2dNative????????!?????1??"L
!model_1/model/conv_dw_3/depthwiseDepthwiseConv2dNative??2???!#l)"j??"N
#model_1/model/conv_dw_1/depthwise_1DepthwiseConv2dNative????????!??????"L
!model_1/model/conv_dw_1/depthwiseDepthwiseConv2dNative-??uP???!_oq?k???"N
#model_1/model/conv_dw_2/depthwise_1DepthwiseConv2dNativeY??t??!??Q9????"L
!model_1/model/conv_dw_2/depthwiseDepthwiseConv2dNative_e?I?f??!???)???"5
model_1/model/conv_pad_2/PadPad?)&??L??!(B ?????"7
model_1/model/conv_pad_2/Pad_1Pad??a ?H??!|P#kӱ??Q      Y@YYi?2??E@a??<?h@L@q
????#@y?&?Rr?"?	
both?Your program is POTENTIALLY input-bound because 62.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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