?	?ihw>l@?ihw>l@!?ihw>l@	dvbEן??dvbEן??!dvbEן??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?ihw>l@J(}!d?`@1=?Е?V@A??$??ܱ?Idv?S???Y???ZD??*	????6c?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2q????;@!??'N?X@),????;@1j???X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@^?zk`???!|>?L5??)?t???m??1??kjm??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@)?'?$???!?/???)?=@??̮?1????t??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@tb?c??!?]4????)tb?c??1?]4????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchC</O???!?>*?????)C</O???1?>*?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismd????t??!gd?????)?ؙB?5??18? v?̣?:Preprocessing2F
Iterator::Model??Ր?Ǫ?!~???c???)????|?r?1?ȃ?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9evbEן??IeH?y'?M@Q%U?8?C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	J(}!d?`@J(}!d?`@!J(}!d?`@      ??!       "	=?Е?V@=?Е?V@!=?Е?V@*      ??!       2	??$??ܱ???$??ܱ?!??$??ܱ?:	dv?S???dv?S???!dv?S???B      ??!       J	???ZD?????ZD??!???ZD??R      ??!       Z	???ZD?????ZD??!???ZD??b      ??!       JGPUYevbEן??b qeH?y'?M@y%U?8?C@?"-
IteratorGetNext/_3_SendEP?????!EP?????"-
IteratorGetNext/_1_Send&???p???!؜?:?a??"L
!model_1/model/conv_dw_3/depthwiseDepthwiseConv2dNative?uM?۝?!5*?????"N
#model_1/model/conv_dw_3/depthwise_1DepthwiseConv2dNative??0?ɝ?!",?6k??"N
#model_1/model/conv_dw_1/depthwise_1DepthwiseConv2dNative???L?F??!F?'???"L
!model_1/model/conv_dw_1/depthwiseDepthwiseConv2dNative3> ?C??!?	??????"L
!model_1/model/conv_dw_2/depthwiseDepthwiseConv2dNative;0p$??!gi????"N
#model_1/model/conv_dw_2/depthwise_1DepthwiseConv2dNative?,`u'???!?7????"5
model_1/model/conv_pad_2/PadPad????!??gX???"7
model_1/model/conv_pad_2/Pad_1Pad(v????!X??M???Q      Y@Y?M?_{D@a>??b??M@q2?#?I@y30^??p?"?	
both?Your program is POTENTIALLY input-bound because 59.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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