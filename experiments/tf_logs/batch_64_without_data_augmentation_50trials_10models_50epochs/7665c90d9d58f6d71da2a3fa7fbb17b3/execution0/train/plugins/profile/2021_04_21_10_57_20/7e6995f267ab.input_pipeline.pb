	-?s?E?t@-?s?E?t@!-?s?E?t@	?!?={???!?={??!?!?={??"n
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
	?T???(@?T???(@!?T???(@      ??!       "	??͕t@??͕t@!??͕t@*      ??!       2      ??!       :	*;??.???*;??.???!*;??.???B      ??!       J	W#??2R??W#??2R??!W#??2R??R      ??!       Z	W#??2R??W#??2R??!W#??2R??b      ??!       JGPUY?!?={??b q?S`?????y???ǋX@