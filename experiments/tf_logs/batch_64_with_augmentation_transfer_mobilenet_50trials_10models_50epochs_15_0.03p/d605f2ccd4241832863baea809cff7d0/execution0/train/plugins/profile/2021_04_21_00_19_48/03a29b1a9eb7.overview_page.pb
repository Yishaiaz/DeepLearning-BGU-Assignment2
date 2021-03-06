?	?֊6G?m@?֊6G?m@!?֊6G?m@	fQ???O??fQ???O??!fQ???O??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?֊6G?m@/??C?a@1?\?	?V@A	À%W???I?FZ*oG??Y????????*	?l??? ?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2W?"??:@!C$R?:?X@)_??,?q:@1?u;RĵX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?o%;6??!?LW*;??)?: ??^??1???????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??o?????!5b??`??)?G??[???1=;s???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?uS?k%??!Y?I?,??)?uS?k%??1Y?I?,??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?`???|??!?H??????)?`???|??1?H??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismmXSYv??!?{ ?Q??)??R%?ޒ?1?]????:Preprocessing2F
Iterator::Model79|҉??!??ۭY???)???W?x?1p??]E??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9fQ???O??IKΨ??N@Q]Vl?x:C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/??C?a@/??C?a@!/??C?a@      ??!       "	?\?	?V@?\?	?V@!?\?	?V@*      ??!       2		À%W???	À%W???!	À%W???:	?FZ*oG???FZ*oG??!?FZ*oG??B      ??!       J	????????????????!????????R      ??!       Z	????????????????!????????b      ??!       JGPUYfQ???O??b qKΨ??N@y]Vl?x:C@?"-
IteratorGetNext/_3_Send=Dg?Z???!=Dg?Z???"-
IteratorGetNext/_1_Send??DR???!;???\???"N
#model_1/model/conv_dw_3/depthwise_1DepthwiseConv2dNative^1??K??!??~????"L
!model_1/model/conv_dw_3/depthwiseDepthwiseConv2dNativeF???:??!R?җ???"L
!model_1/model/conv_dw_1/depthwiseDepthwiseConv2dNative ?X$zۚ?!??1uOI??"N
#model_1/model/conv_dw_1/depthwise_1DepthwiseConv2dNative?f!P?Ě?!_?3:????"L
!model_1/model/conv_dw_2/depthwiseDepthwiseConv2dNativen?nj????!?m?ŕ???"N
#model_1/model/conv_dw_2/depthwise_1DepthwiseConv2dNativeN?<?܈??!LU??????"5
model_1/model/conv_pad_2/PadPad?U+P??!Z??+????"7
model_1/model/conv_pad_2/Pad_1Padd?P?]???!??vڮ??Q      Y@YYi?2??E@a??<?h@L@q???Ĉ?@y??KDq?"?	
both?Your program is POTENTIALLY input-bound because 60.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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