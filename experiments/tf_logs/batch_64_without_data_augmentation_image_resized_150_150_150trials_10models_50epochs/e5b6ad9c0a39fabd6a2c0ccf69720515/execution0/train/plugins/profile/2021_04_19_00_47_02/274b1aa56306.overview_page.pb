?	?7k?g@?7k?g@!?7k?g@	?}??????}?????!?}?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?7k?g@?p]1\Y@1?]?\U@Ap\?M4??I??mnL@YH?V
??*	????.?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?c???H@!KR/?8?X@)????rK@1?????9W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@i??֦???!????H@)???Y???1LZtFx?@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@ٯ;?y???!???̊@)?Ɍ??^??1?d?????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??^f??!;?????)??^f??1;?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?W歺??!Nu:I҂??)?W歺??1Nu:I҂??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???A_z??!?5øP(??)s,??̓?1#?????:Preprocessing2F
Iterator::ModelB?"LQ.??!Ym+??q??)?t?? ?{?1L?AۇK??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?}?????I??&??BK@QK??\?yF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?p]1\Y@?p]1\Y@!?p]1\Y@      ??!       "	?]?\U@?]?\U@!?]?\U@*      ??!       2	p\?M4??p\?M4??!p\?M4??:	??mnL@??mnL@!??mnL@B      ??!       J	H?V
??H?V
??!H?V
??R      ??!       Z	H?V
??H?V
??!H?V
??b      ??!       JGPUY?}?????b q??&??BK@yK??\?yF@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterD?T߯???!D?T߯???0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?<x???!d??]???0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput$?Bk??!??N????0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInputS??????!??.??q??0"@
model/sequential/conv2d_1/Relu_FusedConv2Dn??p???! W/
???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?0r?]??!??????"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3?4F??!???
0??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3????n???!>]c8M??"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter?5l?j???!L???s??0"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!d)RxW???0Q      Y@YZ??]q?@@a? 6QG?P@qH-??!@y??z\tm?"?	
both?Your program is POTENTIALLY input-bound because 53.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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