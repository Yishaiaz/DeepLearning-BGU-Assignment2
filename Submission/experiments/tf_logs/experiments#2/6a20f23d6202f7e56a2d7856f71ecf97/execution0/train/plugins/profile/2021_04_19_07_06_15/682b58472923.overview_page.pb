?	2??|Nh@2??|Nh@!2??|Nh@	??ӯYQ????ӯYQ??!??ӯYQ??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails62??|Nh@?)??%YZ@1bg
?ׇU@A?>U?b??I?đ?@Y?<i????*	??n?=?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2C????p@!?=}???X@)M??StT@1???=?PW@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@W_]????!Hl?\?o@)??Kǜg??1?????@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@l?<*???!?ٸ?p??)l?<*???1?ٸ?p??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@!;oc?#??!?????@)(
??<I??1*???l??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?h>?n??!OK??C??)?h>?n??1OK??C??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismp#e??ݰ?!?'"??W??)?XİØ??1Ul?????:Preprocessing2F
Iterator::Model???L??!4?? ???)qTn???v?1u??Z???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??ӯYQ??I???G?K@Q?Դ?%F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)??%YZ@?)??%YZ@!?)??%YZ@      ??!       "	bg
?ׇU@bg
?ׇU@!bg
?ׇU@*      ??!       2	?>U?b???>U?b??!?>U?b??:	?đ?@?đ?@!?đ?@B      ??!       J	?<i?????<i????!?<i????R      ??!       Z	?<i?????<i????!?<i????b      ??!       JGPUY??ӯYQ??b q???G?K@y?Դ?%F@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter}?N?S???!}?N?S???0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?8p?S???!??_?S???0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInputD??/b??!j?Y??>??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInputOD݅F??!?zlV???0"@
model/sequential/conv2d_1/Relu_FusedConv2D?????Ǣ?!?ReO ??"B
 model/sequential/conv2d_1/Relu_1_FusedConv2DE2?????!??cU??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3?n?(???!?_Hu??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3?C??????!a(?褒??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????4{??!>
?,???0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterJ?0G?n??!?(?p??0Q      Y@YZ??]q?@@a? 6QG?P@q?/U??&@yo???zs?"?

both?Your program is POTENTIALLY input-bound because 54.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?11.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 