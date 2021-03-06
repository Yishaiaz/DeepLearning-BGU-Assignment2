?	t???W?l@t???W?l@!t???W?l@	P?bSd???P?bSd???!P?bSd???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6t???W?l@?ѡ.a@1??ӹ??V@A?K?b???I??RAE???Y]?].?;??*	?l??i??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??4??::@!B@?N?X@)?\???9@1?39?s?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@Tol?`??!?$#{???)?߆?y??1k.?6?7??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@???߽???!:d???)ᛦ????1??????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@???????!??}?????)???????1??}?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchz?sѐ???!?ZJlx???)z?sѐ???1?ZJlx???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism'?|???!?M&?????)L?^I???1?oc????:Preprocessing2F
Iterator::ModelQi??>???!????m???)??v?ӂw?1?n???Y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9P?bSd???I?:??3N@Q??2ߨC@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ѡ.a@?ѡ.a@!?ѡ.a@      ??!       "	??ӹ??V@??ӹ??V@!??ӹ??V@*      ??!       2	?K?b????K?b???!?K?b???:	??RAE?????RAE???!??RAE???B      ??!       J	]?].?;??]?].?;??!]?].?;??R      ??!       Z	]?].?;??]?].?;??!]?].?;??b      ??!       JGPUYP?bSd???b q?:??3N@y??2ߨC@?"-
IteratorGetNext/_1_Send???????!???????"-
IteratorGetNext/_3_Sendi,v?.???!Fb#??"L
!model_1/model/conv_dw_3/depthwiseDepthwiseConv2dNative=߁????!W?N?|h??"N
#model_1/model/conv_dw_3/depthwise_1DepthwiseConv2dNative?3'???!?O?@??"N
#model_1/model/conv_dw_1/depthwise_1DepthwiseConv2dNative??,???!MM????"L
!model_1/model/conv_dw_1/depthwiseDepthwiseConv2dNativeL?F????!???????"L
!model_1/model/conv_dw_2/depthwiseDepthwiseConv2dNativek F5׫??!???\????"N
#model_1/model/conv_dw_2/depthwise_1DepthwiseConv2dNative?)?pO???!?C?1???"7
model_1/model/conv_pad_2/Pad_1Pad?8'dH??!ɕ{?tt??"5
model_1/model/conv_pad_2/PadPad>?XF??!1????^??Q      Y@YYi?2??E@a??<?h@L@qPգ'#@y??,???p?"?	
both?Your program is POTENTIALLY input-bound because 59.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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