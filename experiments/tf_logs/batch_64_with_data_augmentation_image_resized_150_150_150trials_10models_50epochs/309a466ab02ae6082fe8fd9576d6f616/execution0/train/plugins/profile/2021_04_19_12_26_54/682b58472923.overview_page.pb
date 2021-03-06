?	U3k) ?g@U3k) ?g@!U3k) ?g@	h?CnFP??h?CnFP??!h?CnFP??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6U3k) ?g@?x?|Y@1? ?bGU@A???;???I?V???x@YK?*n\??*	+??g?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2a7l[?y@!H?gf??X@)???׺?@1?`?*A?V@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?_?n???!??5@)	kc섗??1ިC?4?@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@???~????!0G!?#???)???~????10G!?#???:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@mT?YO??!c??ҽ@)#?~???1?~_Z????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchg׽?	??!?K??`??)g׽?	??1?K??`??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??;??ز?!?AQ-???)?Ss??P??1?}	Z??:Preprocessing2F
Iterator::Model???4`???!?f????)-σ??v{?1?bs?Yp??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9h?CnFP??I?D????K@Q44?f=F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?x?|Y@?x?|Y@!?x?|Y@      ??!       "	? ?bGU@? ?bGU@!? ?bGU@*      ??!       2	???;??????;???!???;???:	?V???x@?V???x@!?V???x@B      ??!       J	K?*n\??K?*n\??!K?*n\??R      ??!       Z	K?*n\??K?*n\??!K?*n\??b      ??!       JGPUYh?CnFP??b q?D????K@y44?f=F@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??($??!??($??0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterFX?K???!*t?????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput?L]l????!??????0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput??C????!ݕ>0z???0"@
model/sequential/conv2d_1/Relu_FusedConv2D????????!?FY?s\??"B
 model/sequential/conv2d_1/Relu_1_FusedConv2Dv\??Ѣ?!????????"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??jP??!??C>????"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3??Z?P=??!٘g?.??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?*y???!???uZ???0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?*y???!u:?=????0Q      Y@Y?rO#,F@a>?????K@q??A???%@y???X?m?"?

both?Your program is POTENTIALLY input-bound because 53.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?11.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 