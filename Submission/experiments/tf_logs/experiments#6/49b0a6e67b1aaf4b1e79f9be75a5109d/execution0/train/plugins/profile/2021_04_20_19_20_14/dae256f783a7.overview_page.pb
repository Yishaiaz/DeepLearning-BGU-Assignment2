?	?:???m@?:???m@!?:???m@	?O}ݰ????O}ݰ???!?O}ݰ???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?:???m@??"?4a@1??66IX@A?JZ????ILݕ]08??Yal!?A	??*	?rh?-??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2???=z#=@! ?R??X@)c{-???<@1?T共X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??ao??!??X6???)Ҩ??6p??1???????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@N?a??m??!??Q@???)#??~j???1?7oh????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@x%?s}??!?,(v???)x%?s}??1?,(v???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchL??pvk??!a??????)L??pvk??1a??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism#?M)????!i?t??c??)?#??????1_???+???:Preprocessing2F
Iterator::Model⬈?????!? ??[???)?[???u?1?Ik?)???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?O}ݰ???I!??YgM@Q?f:?ކD@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??"?4a@??"?4a@!??"?4a@      ??!       "	??66IX@??66IX@!??66IX@*      ??!       2	?JZ?????JZ????!?JZ????:	Lݕ]08??Lݕ]08??!Lݕ]08??B      ??!       J	al!?A	??al!?A	??!al!?A	??R      ??!       Z	al!?A	??al!?A	??!al!?A	??b      ??!       JGPUY?O}ݰ???b q!??YgM@y?f:?ކD@?"-
IteratorGetNext/_1_SendMJU~????!MJU~????"-
IteratorGetNext/_4_Recv?g????!4??6:???"L
!model_1/model/conv_dw_3/depthwiseDepthwiseConv2dNative??	>i??!?:?y??"N
#model_1/model/conv_dw_3/depthwise_1DepthwiseConv2dNative??y?Ub??!??r?/??"L
!model_1/model/conv_dw_1/depthwiseDepthwiseConv2dNative???'???!?_%B???"N
#model_1/model/conv_dw_1/depthwise_1DepthwiseConv2dNative?zM
|??!????Q??"L
!model_1/model/conv_dw_2/depthwiseDepthwiseConv2dNative?؀??!????5??"N
#model_1/model/conv_dw_2/depthwise_1DepthwiseConv2dNative#/???{??!E8bU???"5
model_1/model/conv_pad_2/PadPad??s???!jh?????"7
model_1/model/conv_pad_2/Pad_1Pad? U$-??!Vn????Q      Y@Y?M?_{D@a>??b??M@q??k?*`@y։g??m?"?	
both?Your program is POTENTIALLY input-bound because 58.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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