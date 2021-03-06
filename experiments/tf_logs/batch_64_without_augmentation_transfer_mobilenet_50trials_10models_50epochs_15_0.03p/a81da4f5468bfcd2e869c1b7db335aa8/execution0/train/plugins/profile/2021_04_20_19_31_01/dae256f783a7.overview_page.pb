?	`vOKk@`vOKk@!`vOKk@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-`vOKk@wj.7??`@1?????$U@A??9?ؗ??I??U???*	NbX??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?pt??&6@!??	?x?X@)??L??5@1py???X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@5?;???!???c???){?\?&???1???o?:??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??!p??!?"?????)???_?5??1???@~??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??RB????!'?	?????)??RB????1'?	?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch׈`\:??!? wgU??)׈`\:??1? wgU??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism؀q????!䉖]b???)??R%ʎ?1??k??R??:Preprocessing2F
Iterator::Model?K??$w??!?o ?G???)иp $x?1?]??Y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI? ?_ݡN@Q?4?"^C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	wj.7??`@wj.7??`@!wj.7??`@      ??!       "	?????$U@?????$U@!?????$U@*      ??!       2	??9?ؗ????9?ؗ??!??9?ؗ??:	??U?????U???!??U???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q? ?_ݡN@y?4?"^C@?"-
IteratorGetNext/_4_Recv??ĕ+[??!??ĕ+[??"-
IteratorGetNext/_2_Recv??M??K??!?C	,?S??"L
!model_1/model/conv_dw_3/depthwiseDepthwiseConv2dNative???_٤??!?kXH??"N
#model_1/model/conv_dw_3/depthwise_1DepthwiseConv2dNativey88?????!n9?4???"L
!model_1/model/conv_dw_1/depthwiseDepthwiseConv2dNative???N(???!7???????"N
#model_1/model/conv_dw_1/depthwise_1DepthwiseConv2dNative?a???Ɯ?!S?m9/???"N
#model_1/model/conv_dw_2/depthwise_1DepthwiseConv2dNative?L??f??!??4????"L
!model_1/model/conv_dw_2/depthwiseDepthwiseConv2dNative7??\Y??!???1???"5
model_1/model/conv_pad_2/PadPad???8?B??!_=??E???"7
model_1/model/conv_pad_2/Pad_1Pad?t???=??!Ay5???Q      Y@YYi?2??E@a??<?h@L@q?,c?9"@y 4?i?3q?"?	
both?Your program is POTENTIALLY input-bound because 60.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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