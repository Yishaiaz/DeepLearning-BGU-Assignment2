?	W???:m@W???:m@!W???:m@	"ƫ?m??"ƫ?m??!"ƫ?m??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6W???:m@?a??m?a@1x?a??pV@Ax` ?C???I?O:?`*@Y??-$`??*	?G??/?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV21?q?	?8@!%Ʉ??X@)?r?]?x8@1E????X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@;U?g$B??!?p?Ȉ??)???×??1l?و????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@aR||Bv??!r?x???)%̴?+??1?,h?GZ??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@v?X????!,z?v???)v?X????1,z?v???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?D.8????!=???????)?D.8????1=???????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??=^H???!j}??g???)y=???1?t.t8j??:Preprocessing2F
Iterator::ModelT?qs*??!J?6{?S??)w-!?lv?1P?Q.???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9"ƫ?m??I?(???N@Q?^G;?1C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?a??m?a@?a??m?a@!?a??m?a@      ??!       "	x?a??pV@x?a??pV@!x?a??pV@*      ??!       2	x` ?C???x` ?C???!x` ?C???:	?O:?`*@?O:?`*@!?O:?`*@B      ??!       J	??-$`????-$`??!??-$`??R      ??!       Z	??-$`????-$`??!??-$`??b      ??!       JGPUY"ƫ?m??b q?(???N@y?^G;?1C@?"-
IteratorGetNext/_2_Recv???I^??!???I^??"-
IteratorGetNext/_4_Recv?
?IS???!*mr??"N
#model_1/model/conv_dw_3/depthwise_1DepthwiseConv2dNative?蒸:???!e¢???"L
!model_1/model/conv_dw_3/depthwiseDepthwiseConv2dNative?ľj???!???N????"L
!model_1/model/conv_dw_1/depthwiseDepthwiseConv2dNative_??l/??!???????"N
#model_1/model/conv_dw_1/depthwise_1DepthwiseConv2dNative_??l/??!?df?{]??"N
#model_1/model/conv_dw_2/depthwise_1DepthwiseConv2dNative?{??????!?X?i?T??"L
!model_1/model/conv_dw_2/depthwiseDepthwiseConv2dNative??M?͎?!r??_K??"5
model_1/model/conv_pad_2/PadPadU?ҏ???!?۪bd7??"7
model_1/model/conv_pad_2/Pad_1Padځ?? j??!???g?"??Q      Y@YYi?2??E@a??<?h@L@q????0"@y??0?>5p?"?	
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