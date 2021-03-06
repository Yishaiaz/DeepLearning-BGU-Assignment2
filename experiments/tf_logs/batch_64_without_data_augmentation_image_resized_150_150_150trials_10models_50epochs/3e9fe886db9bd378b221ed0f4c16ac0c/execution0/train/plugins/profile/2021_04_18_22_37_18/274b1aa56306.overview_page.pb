?	?$"???h@?$"???h@!?$"???h@	oQ?????oQ?????!oQ?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?$"???h@O??D?`Z@1?st?V@AyY??I>?>tA?@YG?&ji???*	???MB??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?d8???@!??
0C?X@)?-II?@1{W?^@W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?p?Qe??!??>?G.@)??????1???6Q@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@>?h??!?s?g????)>?h??1?s?g????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@Tƿϸ??!X_#Y@)?|A	??1Q<1???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???T???!Bw#????)???T???1Bw#????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??-???!???EUo??)?蹅?D??1
?H?Ok??:Preprocessing2F
Iterator::Model.??M?Ұ?!C?3/??)JB"m?Ot?1#?y????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9nQ?????I@?r??&K@Q?????F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	O??D?`Z@O??D?`Z@!O??D?`Z@      ??!       "	?st?V@?st?V@!?st?V@*      ??!       2	yY??yY??!yY??:	>?>tA?@>?>tA?@!>?>tA?@B      ??!       J	G?&ji???G?&ji???!G?&ji???R      ??!       Z	G?&ji???G?&ji???!G?&ji???b      ??!       JGPUYnQ?????b q@?r??&K@y?????F@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter4??mԻ??!4??mԻ??0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterخKpW???!??????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInputw??{?C??!u??^?#??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput???ѹ:??!N?????0"@
model/sequential/conv2d_1/Relu_FusedConv2DX??}???!9?!;S???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D͠@?????!S?I?'/??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV3?????\??!????:??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??u5yR??!#fN?E??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??۶???!o????0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter+??????!???r]??0Q      Y@YZ??]q?@@a? 6QG?P@q?'|?!@y??O?k?"?	
both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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