# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.proto',
  package='mediapipe.tasks.vision.image_segmenter.proto',
  syntax='proto2',
  serialized_options=b'\n6com.google.mediapipe.tasks.vision.imagesegmenter.protoB\025SegmenterOptionsProto',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\nGmediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.proto\x12,mediapipe.tasks.vision.image_segmenter.proto\"\xdf\x02\n\x10SegmenterOptions\x12m\n\x0boutput_type\x18\x01 \x01(\x0e\x32I.mediapipe.tasks.vision.image_segmenter.proto.SegmenterOptions.OutputType:\rCATEGORY_MASK\x12\x63\n\nactivation\x18\x02 \x01(\x0e\x32I.mediapipe.tasks.vision.image_segmenter.proto.SegmenterOptions.Activation:\x04NONE\"E\n\nOutputType\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\x11\n\rCATEGORY_MASK\x10\x01\x12\x13\n\x0f\x43ONFIDENCE_MASK\x10\x02\"0\n\nActivation\x12\x08\n\x04NONE\x10\x00\x12\x0b\n\x07SIGMOID\x10\x01\x12\x0b\n\x07SOFTMAX\x10\x02\x42O\n6com.google.mediapipe.tasks.vision.imagesegmenter.protoB\x15SegmenterOptionsProto'
)



_SEGMENTEROPTIONS_OUTPUTTYPE = _descriptor.EnumDescriptor(
  name='OutputType',
  full_name='mediapipe.tasks.vision.image_segmenter.proto.SegmenterOptions.OutputType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='CATEGORY_MASK', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='CONFIDENCE_MASK', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=354,
  serialized_end=423,
)
_sym_db.RegisterEnumDescriptor(_SEGMENTEROPTIONS_OUTPUTTYPE)

_SEGMENTEROPTIONS_ACTIVATION = _descriptor.EnumDescriptor(
  name='Activation',
  full_name='mediapipe.tasks.vision.image_segmenter.proto.SegmenterOptions.Activation',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SIGMOID', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SOFTMAX', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=425,
  serialized_end=473,
)
_sym_db.RegisterEnumDescriptor(_SEGMENTEROPTIONS_ACTIVATION)


_SEGMENTEROPTIONS = _descriptor.Descriptor(
  name='SegmenterOptions',
  full_name='mediapipe.tasks.vision.image_segmenter.proto.SegmenterOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='output_type', full_name='mediapipe.tasks.vision.image_segmenter.proto.SegmenterOptions.output_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='activation', full_name='mediapipe.tasks.vision.image_segmenter.proto.SegmenterOptions.activation', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SEGMENTEROPTIONS_OUTPUTTYPE,
    _SEGMENTEROPTIONS_ACTIVATION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=122,
  serialized_end=473,
)

_SEGMENTEROPTIONS.fields_by_name['output_type'].enum_type = _SEGMENTEROPTIONS_OUTPUTTYPE
_SEGMENTEROPTIONS.fields_by_name['activation'].enum_type = _SEGMENTEROPTIONS_ACTIVATION
_SEGMENTEROPTIONS_OUTPUTTYPE.containing_type = _SEGMENTEROPTIONS
_SEGMENTEROPTIONS_ACTIVATION.containing_type = _SEGMENTEROPTIONS
DESCRIPTOR.message_types_by_name['SegmenterOptions'] = _SEGMENTEROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SegmenterOptions = _reflection.GeneratedProtocolMessageType('SegmenterOptions', (_message.Message,), {
  'DESCRIPTOR' : _SEGMENTEROPTIONS,
  '__module__' : 'mediapipe.tasks.cc.vision.image_segmenter.proto.segmenter_options_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.tasks.vision.image_segmenter.proto.SegmenterOptions)
  })
_sym_db.RegisterMessage(SegmenterOptions)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
