?	-?s?E?t@-?s?E?t@!-?s?E?t@	?!?={???!?={??!?!?={??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails--?s?E?t@?T???(@1??͕t@I*;??.???YW#??2R??*	??ʑ6?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?iN^d?)@!????&?X@)???#??(@1?V 5./X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???I'??!T?4_@)?1Xq????1??>???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??,?"??!)?026??)??,?"??1)?026??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@ė?"?n??!?LefK???){?????1????d
??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchѕT? ??!D?{{?L??)ѕT? ??1D?{{?L??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism
K<?lʱ?!QI??:??)? ?س???1?fpPN??:Preprocessing2F
Iterator::Model??O9&???!?#??????)?6??|?1s?M+(??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?!?={??I?S`?????Q???ǋX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?T???(@?T???(@!?T???(@      ??!       "	??͕t@??͕t@!??͕t@*      ??!       2      ??!       :	*;??.???*;??.???!*;??.???B      ??!       J	W#??2R??W#??2R??!W#??2R??R      ??!       Z	W#??2R??W#??2R??!W#??2R??b      ??!       JGPUY?!?={??b q?S`?????y???ǋX@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?v_????!?v_????0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????e??!??[???0"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?b?#???!R??B?*??"@
model/sequential/conv2d_1/Relu_FusedConv2D???r???!?h???N??"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput???:Ql??!N??	????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput)U?dCg??!?ZPvށ??0"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3?"xoq??!L_?????"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3U1f`??!7?? ???"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterS}?????!???????0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterM61?????!?֡&A??0Q      Y@Y?e?Gʪ=@a?&nM?Q@q?nOvS?@yU??TQ?"?	
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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