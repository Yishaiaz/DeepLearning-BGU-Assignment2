?	7erl@7erl@!7erl@	ƪE??"??ƪE??"??!ƪE??"??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails67erl@&r??a@1??Z}?U@A??im۳?IRew?@Y?F<????*	+???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2X??G@!?8s??X@)?R?r/p@1?????~V@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@u><K???!?Fgy?@)/?????1U?Z)?@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@^-wf????!8?d?e @)??u6????1?R^?n?@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@u??<???!?+??w??)u??<???1?+??w??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?XP?i??!ѻw?)??)?XP?i??1ѻw?)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????6???!I??????)??M~?N??1??P.07??:Preprocessing2F
Iterator::Model???:8ط?!`?1c~??)[?a/?}?1?p/iC??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ƪE??"??I?K?$??N@Q?(MD?AC@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	&r??a@&r??a@!&r??a@      ??!       "	??Z}?U@??Z}?U@!??Z}?U@*      ??!       2	??im۳???im۳?!??im۳?:	Rew?@Rew?@!Rew?@B      ??!       J	?F<?????F<????!?F<????R      ??!       Z	?F<?????F<????!?F<????b      ??!       JGPUYƪE??"??b q?K?$??N@y?(MD?AC@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter冾W???!冾W???0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter???????!]D?????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput_?7&tm??!i;???:??0"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput>??̫_??!?????0"@
model/sequential/conv2d_1/Relu_FusedConv2Dhx#?侢?![??u???"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D?c??????!???,?S??"q
Ggradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3\ 0?u???!????k??"s
Igradient_tape/model/sequential/batch_normalization/FusedBatchNormGradV3_1FusedBatchNormGradV32?z????!??ݖ[???"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?)	????!?=?C????0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterM:ϟ???!??0B p??0Q      Y@YZ??]q?@@a? 6QG?P@q????10@y?F???#s?"?

both?Your program is POTENTIALLY input-bound because 60.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 