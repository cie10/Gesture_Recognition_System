# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/framework/calculator_options.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='mediapipe/framework/calculator_options.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=b'\n\032com.google.mediapipe.protoB\026CalculatorOptionsProto',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n,mediapipe/framework/calculator_options.proto\x12\tmediapipe\"9\n\x11\x43\x61lculatorOptions\x12\x18\n\x0cmerge_fields\x18\x01 \x01(\x08\x42\x02\x18\x01*\n\x08\xa0\x9c\x01\x10\x80\x80\x80\x80\x02\x42\x34\n\x1a\x63om.google.mediapipe.protoB\x16\x43\x61lculatorOptionsProto'
)




_CALCULATOROPTIONS = _descriptor.Descriptor(
  name='CalculatorOptions',
  full_name='mediapipe.CalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='merge_fields', full_name='mediapipe.CalculatorOptions.merge_fields', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\030\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(20000, 536870912), ],
  oneofs=[
  ],
  serialized_start=59,
  serialized_end=116,
)

DESCRIPTOR.message_types_by_name['CalculatorOptions'] = _CALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CalculatorOptions = _reflection.GeneratedProtocolMessageType('CalculatorOptions', (_message.Message,), {
  'DESCRIPTOR' : _CALCULATOROPTIONS,
  '__module__' : 'mediapipe.framework.calculator_options_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.CalculatorOptions)
  })
_sym_db.RegisterMessage(CalculatorOptions)


DESCRIPTOR._options = None
_CALCULATOROPTIONS.fields_by_name['merge_fields']._options = None
# @@protoc_insertion_point(module_scope)
