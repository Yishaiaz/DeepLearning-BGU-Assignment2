?	???Of@???Of@!???Of@	2??Yp1??2??Yp1??!2??Yp1??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???Of@-?B;??@1x??D??e@I?pvk?L??Y-???????*	??n??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2jܛ?0?'@!sZ)^??X@)؟??N '@1p?p?HX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@K???>??!i +?]@)?tۈ??1???(??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??st??!w????{??)?FZ*oG??1@jO?i??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?Nϻ????!???O4???)?Nϻ????1???O4???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchDOʤ???!?ǈ?V??)DOʤ???1?ǈ?V??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismLP÷?n??!+?l؋???)?nJy???1???7????:Preprocessing2F
Iterator::Model?V]?jJ??!c?R?Ј??)nj????}?1??].Qd??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no92??Yp1??I???ȁ?@Q??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-?B;??@-?B;??@!-?B;??@      ??!       "	x??D??e@x??D??e@!x??D??e@*      ??!       2      ??!       :	?pvk?L???pvk?L??!?pvk?L??B      ??!       J	-???????-???????!-???????R      ??!       Z	-???????-???????!-???????b      ??!       JGPUY2??Yp1??b q???ȁ?@y??X@?"o
Cgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter9??????!9??????0"q
Egradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFilter??'???!???????0"@
model/sequential/conv2d_1/Relu_FusedConv2DS??????!?_BNt??"B
 model/sequential/conv2d_1/Relu_1_FusedConv2D??? T???!+6L??*??"m
Bgradient_tape/model/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?;ؾ^???!?='}????0"o
Dgradient_tape/model/sequential/conv2d_1/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput???x???!?ޜ???0"-
IteratorGetNext/_1_Send?,?p,C??!??*R@??"m
Agradient_tape/model/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterO??@????!Q<?R??0"o
Cgradient_tape/model/sequential/conv2d/Conv2D_1/Conv2DBackpropFilterConv2DBackpropFiltery???e???!??p??c??0"o
Cgradient_tape/model/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterԸ?5???!??ߣ??0Q      Y@Ye??/vA@aM*}?DP@q???E|?
@y???5?9]?"?	
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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