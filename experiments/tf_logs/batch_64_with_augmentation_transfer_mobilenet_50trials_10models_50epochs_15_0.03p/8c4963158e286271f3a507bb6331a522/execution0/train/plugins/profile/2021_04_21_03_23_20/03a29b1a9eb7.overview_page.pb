?	?p???_n@?p???_n@!?p???_n@	qv?i؁??qv?i؁??!qv?i؁??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?p???_n@%Z?xZ?b@1:tzލ?V@AL8????INA~6?@YVҊo(|??*???K??@)      ?=2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2fh<ĩ:@!1t?_??X@)?K?p:@1һy???X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@????????!=/?Bj???)??^???1E???H??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@8K?r??!7??vG??);?I/???1ϼjǽA??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?Ǚ&l???!?ԟ^???)?Ǚ&l???1?ԟ^???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch>?*??!26)í???)>?*??126)í???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismSB??^~??!??ڇt??)S???.Q??1g??^hk??:Preprocessing2F
Iterator::Modelʥ??$??!>ϋ ???)J?i?Wv?1?[?1????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 61.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9qv?i؁??I???rOO@Qv??Q??B@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	%Z?xZ?b@%Z?xZ?b@!%Z?xZ?b@      ??!       "	:tzލ?V@:tzލ?V@!:tzލ?V@*      ??!       2	L8????L8????!L8????:	NA~6?@NA~6?@!NA~6?@B      ??!       J	VҊo(|??VҊo(|??!VҊo(|??R      ??!       Z	VҊo(|??VҊo(|??!VҊo(|??b      ??!       JGPUYqv?i؁??b q???rOO@yv??Q??B@?"-
IteratorGetNext/_4_RecvO~??7"??!O~??7"??"-
IteratorGetNext/_1_SendL??r1???!?s9`Л??"L
!model_1/model/conv_dw_3/depthwiseDepthwiseConv2dNative?Ѵ ?W??!(0`???"N
#model_1/model/conv_dw_3/depthwise_1DepthwiseConv2dNative??3z?L??!B˧,x??"L
!model_1/model/conv_dw_1/depthwiseDepthwiseConv2dNative6Ӡ?՚?!nuء?%??"N
#model_1/model/conv_dw_1/depthwise_1DepthwiseConv2dNative
#?ڞϚ?!???????"N
#model_1/model/conv_dw_2/depthwise_1DepthwiseConv2dNativeym?'3???!#?(}???"L
!model_1/model/conv_dw_2/depthwiseDepthwiseConv2dNativeX??L???!?5ɏ????"5
model_1/model/conv_pad_2/PadPad??݄?"??!?%??????"7
model_1/model/conv_pad_2/Pad_1Pad5?z9???!M??m????Q      Y@YYi?2??E@a??<?h@L@q?d H?@y3{H+U+p?"?	
both?Your program is POTENTIALLY input-bound because 61.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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