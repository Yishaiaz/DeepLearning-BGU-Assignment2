?	??B=?e@??B=?e@!??B=?e@	???ǚ??????ǚ???!???ǚ???"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??B=?e@??2P@1?_?5?/e@IqX?Q??Y7???????*	?v??
?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?X??i'@!?p???X@)??Û5?&@1NT?JFX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@ C?*??!??#?? @)3?뤾,??1
? bY???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch ??q????!+{ܫ??) ??q????1+{ܫ??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??0???!?{?/Y??)??0???1?{?/Y??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?Q<????!??6????)??;jL???1??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???,յ?!????"??)??4)ݎ?19G?'5Z??:Preprocessing2F
Iterator::Model$?@??!U??(?q??)?7L4H?s?1?I0???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???ǚ???I ??3~@Q???s?/X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??2P@??2P@!??2P@      ??!       "	?_?5?/e@?_?5?/e@!?_?5?/e@*      ??!       2      ??!       :	qX?Q??qX?Q??!qX?Q??B      ??!       J	7???????7???????!7???????R      ??!       Z	7???????7???????!7???????b      ??!       JGPUY???ǚ???b q ??3~@y???s?/X@?"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilterD?@ֆ??!D?@ֆ??0"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter%Ii?????!?"UKۄ??0"@
model/sequential/conv2d_1/Relu_FusedConv2D?%)y??!????%c??"B
 model/sequential/conv2d_1/Relu_1_FusedConv2DUG޼g??!f??a???"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput^??????!??
_????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput^B??	~??!?G?????0"-
IteratorGetNext/_1_Send?3e????!VX???<??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterlVT?kܠ?!$???X??0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter???à?!??֬?p??0"o
Cgradient_tape/model/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ѕ?4???!?`{{??0Q      Y@Ye??/vA@aM*}?DP@q8 n??@yjc??Ա]?"?	
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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