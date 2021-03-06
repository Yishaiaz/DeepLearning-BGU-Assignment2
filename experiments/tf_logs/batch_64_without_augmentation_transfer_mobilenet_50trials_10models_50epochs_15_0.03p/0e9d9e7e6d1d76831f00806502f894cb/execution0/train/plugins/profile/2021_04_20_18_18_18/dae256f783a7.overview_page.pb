?	?7?{?)l@?7?{?)l@!?7?{?)l@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?7?{?)l@??I??!`@1??Đ??W@AF?̱????I%??CK??*	z?&1???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??Z}u99@!"?,.?X@)?.n??8@1nߌ?q?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@" 8?????!?YxI:^??)TpxADj??1¨ Io??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??{?ʄ??!B褂P??)?Fˁj??1????-0??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@h?XR?>??!sKd?????)h?XR?>??1sKd?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?}s??!??DDK???)?}s??1??DDK???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismA?)V°?!]h?Q
???)?$[]N	??1????????:Preprocessing2F
Iterator::Modelˢ?????!S?/?????)??lXSYt?1m_????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?E??qM@Q??,#??D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??I??!`@??I??!`@!??I??!`@      ??!       "	??Đ??W@??Đ??W@!??Đ??W@*      ??!       2	F?̱????F?̱????!F?̱????:	%??CK??%??CK??!%??CK??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?E??qM@y??,#??D@?"-
IteratorGetNext/_4_Recv#}?9????!#}?9????"-
IteratorGetNext/_2_Recv?QXHܵ?! &?e"???"N
#model_1/model/conv_dw_3/depthwise_1DepthwiseConv2dNative?q??9???!-???????"L
!model_1/model/conv_dw_3/depthwiseDepthwiseConv2dNativeEc?R???!a????t??"L
!model_1/model/conv_dw_1/depthwiseDepthwiseConv2dNativev??Z??!H?Ę_&??"N
#model_1/model/conv_dw_1/depthwise_1DepthwiseConv2dNative.?
i?
??!?U????"5
model_1/model/conv_pad_2/PadPadF\Е	??!?(lT???"7
model_1/model/conv_pad_2/Pad_1Pad???/????!*f?????"N
#model_1/model/conv_dw_2/depthwise_1DepthwiseConv2dNative(?-*???!kW??»??"L
!model_1/model/conv_dw_2/depthwiseDepthwiseConv2dNative 3ɉ3|??!?>s????Q      Y@YYi?2??E@a??<?h@L@q?? ?^!@y?0??'Bp?"?	
both?Your program is POTENTIALLY input-bound because 57.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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