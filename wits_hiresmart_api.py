# -*- coding: utf-8 -*-
"""
wits_hiresmart_api
------------------

Default plugin to serve the REpresentational State Transfer (REST) Application Programming Interface (API) for the ``wits_hiresmart`` system.

.. digraph:: wits_hiresmart_api

    forcelabels=true;
    rankdir=LR;

    plugin [label="wits_hiresmart_api\\nWebApplicationPlugin" shape=component];
    database [label="Database" shape="cylinder"];
    user [shape=oval];
    model [shape=box3d];
    config [label="host = 'localhost'\\l|port = 8080\\l|upload_url = '/api/upload'\\l|search_url = '/api/search\\l|similar_words_url = 'api/similar'\\l|rank_url = '/api/rank'\\l" shape=record];

    user -> plugin [label=" IN: upload, search\\n rank, similar  "];
    plugin -> user [label=" OUT: resumes, words  "];

    plugin -> database [label=" OUT: upload  "];
    database -> plugin [label=" IN: resumes  "];
    config -> plugin [label="config.toml"];

    plugin -> model [label=" IN: similar  "];
    model -> plugin [label=" OUT: words  "];
|

**API Paths**

Assuming default configuration, the following API paths are used:

* ``/api/upload``: upload a resume through the ``file[]`` body form input
* ``/api/download``: download a resume with a unique file identifier (see /api/search response)
* ``/api/search``: search through parsed resume data
* ``/api/rank``: search and rank resumes based on word and entity relevance
* ``/api/similar``: get similar words learned from resumes for a set of keywords

The general response of the API contains the following JSON data:

.. jupyter-execute::
    :hide-code:

    from pprint import pprint
    response = dict(
        copyright = 'Copyright description',
        status = 200,
        status_type = 'OK',
        message = 'A description for the status'
    )
    pprint(response)

* ``copyright`` (text): a copyright text message, only displayed on successful request status ``200``
* ``status`` (numeric): the status code of the response, one of ``200``, ``202``, ``400``, ``404``, or ``500``
* ``status_type`` (text): the status type, one of ``OK``, ``ERROR`` or ``UNAVAILABLE``
* ``message`` (text): a description for the status

**Search API Response**

The search API responds with JSON data, containing the following example fields for a successful request:

.. jupyter-execute::
    :hide-code:

    from pprint import pprint
    response = dict(
        copyright = '© 2020 WITS Consulting Inc.',
        status = 200,
        status_type = 'OK',
        message = '2 records returned!',
        data = [
            {
                'text': 'resume text...',
                'first_name': 'Joe',
                'upload_date': '2020-02-29T17:44:15.909981-05:00',
                'original_file_name': 'joe_resume.txt',
                'degrees': 1,
                'download_link': '/api/download/83218321.txt'
            },
            {
                'text': 'resume 2 text...',
                'first_name': 'Lee',
                'upload_date': '2020-03-01T18:44:15.909981-05:00',
                'original_file_name': 'lee_resume.docx',
                'degrees': 0,
                'download_link': '/api/download/8095389321.docx'
            }
        ]
    )
    pprint(response)

* ``data`` (JSON array): a list of JSON data with parsed data fields for each resume, these fields are the same as the non-reserved query string parameters (see **Search API Parameters**)
    * ``text`` (text): the parsed text in the resume
    * ``upload_date`` (text): the date and time in UTC ISO 8601 format in the Eastern timezone
    * ``original_file_name`` (text): the original name of the file that was uploaded
    * ``download_link`` (text): the download link if ``download`` is ``true`` in the config, this is added to each item in the ``data`` JSON list

**Search API Parameters**

For the search API, the following query string parameters are available:

.. jupyter-execute::
    :hide-code:

    from pandas.api.types import is_numeric_dtype
    from wits_resume_data import read_sample
    from wits_resume_parser import parse_to_dataframe

    # Get sample resume parsed
    sample_resume = read_sample('resume_sample.txt')
    parse_sample = parse_to_dataframe(sample_resume)

    # Get columns and types
    columns = parse_sample.columns.to_list()
    column_types = [parse_sample[c] for c in columns]
    column_types = ['numeric' if is_numeric_dtype(c) else 'text' for c in column_types]

    # Print out column and types
    for c, t in zip(columns, column_types):
        print('{} ({})'.format(c, t))

Other reserved search API parameters are available (these cannot be used with operators):

* ``limit``: the number of records or resumes to return
* ``only``: only return the certain fields in the resume data, separated with a comma ``,`` for each field
* ``case_sensitive``: set to ``true`` to make text searches case sensitive and ``false`` to make case insensitive

Examples: 

* Search up to 10 resumes with first name "Joe" ``/api/search?first_name=Joe&degrees_gt=0&limit=10``
* Search resumes with first name "Joe" but only return the "text" field ``/api/search?first_name=Joe&only=text``

**Search API Operators**

To perform operations for the search API, add the operator after an underscore to the end of the parameter:

.. jupyter-execute::
    :hide-code:

    from wits_hiresmart.default_plugins.wits_hiresmart_api import __operators__
    from pprint import pprint

    pprint(__operators__)

Examples:

* Greater than (``>``) for parameter ``degrees`` and value ``0`` = ``degrees_gt=0``
* Similar (``SIMILAR TO``) for parameter ``text`` and value ``%java%`` = ``text_similar=%java%``
* Search all resumes with at least one degree with word "java" ``/api/search?text_similar=%java%&degrees_gt=0``

**Rank API Response**

The rank API responds with JSON data, containing the following example fields for a successful request:

.. jupyter-execute::
    :hide-code:

    from pprint import pprint
    response = dict(
        copyright = '© 2020 WITS Consulting Inc.',
        status = 200,
        status_type = 'OK',
        message = '2 records returned!',
        data = [
            {
                'text': 'resume text...',
                'first_name': 'Joe',
                'degrees': 1,
                'rank': 1,
                'score': 100
            },
            {
                'text': 'resume 2 text...',
                'first_name': 'Lee',
                'degrees': 0,
                'rank': 2,
                'score': 67
            }
        ]
    )
    pprint(response)

* ``data`` (JSON array): a list of JSON data with parsed data fields, scores, these fields are the same as the search API response with additional rank and score fields

The following rank and score fields are added in addition to the search API response fields:

.. jupyter-execute::
    :hide-code:

    from wits_resume_ranker import score_functions, calculate_scores

    points = {k + '_points': 1 for k in score_functions}
    rank_data = calculate_scores(keywords = ['a'], text = ['a resume file'], model = './models/word_vector', **points)
    
    for d in rank_data.columns.to_list():
        print(d)

These added fields are conditional upon which API parameters are passed.
For example, if ``word_points`` are passed, the only ``word_...`` fields are added. 

**Rank API Parameters**

For the rank API, the following query string parameters are available for assigning scoring criteria:

.. jupyter-execute::
    :hide-code:

    from wits_resume_ranker import score_functions

    for s in score_functions:
        print('{}_points'.format(s))

Point parameters can be any number, and are used to weigh each scoring criteria based on user preferences.
For example, if ``word_points`` is given 90 points, and ``skill_points`` is given 10 points, then resumes will be
scored primarily on word relevance (90 points max for ``word_points``), and skill relevance has only a minor effect (10 points max for ``skill_points``).

Other reserved rank API parameters are available (these cannot be used with operators):

* ``keywords`` (*rank API only*): the keywords to base score and ranking on, separated with a comma ``,``
* Any other parameter from the search API

Examples:

* Score developer resumes with max score of 100 ``/api/rank?keywords=java,developer&word_points=100``
* Get only top 10 candidates ``/api/rank?keywords=java,developer&word_points=100&top=10``
* Only score candidates with at least 1 degree ``/api/rank?keywords=java,developer&word_points=100&degrees_gt=0``

**Similarity API Response**

The word similarity API responds with JSON data, containing the following example fields for a successful request:

.. jupyter-execute::
    :hide-code:

    from pprint import pprint
    response = dict(
        copyright = '© 2020 WITS Consulting Inc.',
        status = 200,
        status_type = 'OK',
        message = '2 records returned!',
        data = [
            {
                'word': 'html',
                'similarity': 0.99321
            },
            {
                'word': 'js',
                'similarity': 0.98327
            }
        ]
    )
    pprint(response)

* ``data`` (JSON array): a list of JSON data with similar words to the keywords and their similarity score from 0 to 1

**Similar API Parameters**

For the word similarity API, the following parameters are available:

* ``keywords``: keywords to get similar words for separated with a comma ``,``
* ``top``: the number of most similar words to return, default of 10

Examples:

* Get similar words for java ``/api/similar?keywords=java``
* Get similar words for java and python ``/api/similar?keywords=java,python``
* Get the top 30 most similar words for software ``/api/similar?keywords=software&top=30``

Plugin Parameters
-----------------
name (str or "wits_hiresmart_api")
    Unique name of the plugin to identify it in the ``config.toml``.
kind (str or "WebApplicationPlugin")
    The kind of plugin. See :mod:`wits_hiresmart.plugin`.
parameters (dict)
    The dictionary of parameters passed to the ``run`` function.

    * ``copyright`` (str or '© 2020 WITS Consulting Inc.'): copyright message to include in outputs
    * ``host`` (str or 'localhost'): the host name to serve the API at
    * ``port`` (int or 8080): the port number to serve th API at
    * ``data_format`` (str or 'records'): the data format to use (see :meth:`pandas:pandas.DataFrame.to_dict`)
    * ``words_model`` (str or './models/word_vector'): the path to the word vector model file (see :mod:`wits_hiresmart.default_plugins.word_vector_model`)
    * ``upload`` (bool or True): whether to enable the upload API or not
    * ``upload_url`` (str or '/api/upload'): the upload API url path
    * ``upload_folder`` (str or './uploads'): the folder on the server to store the uploaded resume files
    * ``upload_table`` (str or 'resumes'): the database table name to store the uploaded resume data in
    * ``upload_date_column`` (str or 'date'): the database column to store the date and time for the uploaded resumes
    * ``upload_text_column`` (str or 'text'): the database column to store the plain text from the uploaded resumes
    * ``upload_file_column`` (str or 'file_path'): the database column to store the file path to the uploaded resume files
    * ``similar_words`` (bool or True): whether to enable the similar words API or not
    * ``similar_words_url`` (str or '/api/similar'): the similar words API url path
    * ``similar_words_data_format`` (str or 'records'): same as ``data_format`` but for the similar words API only
    * ``search`` (bool or True): whether to enable the search API or not
    * ``search_data_format`` (str or 'records'): same as ``data_format`` but for the search API only
    * ``search_url`` (str or '/api/search'): the search API url path
    * ``search_limit`` (int or None): a limit to the number of records returned from searches, if default then all records can be returned
    * ``search_columns`` (list(str) or None): list of columns that are searchable through the search API or if default then all columns can be searched
    * ``search_restricted_columns`` (list(str) or ['file_path']): list of columns to restrict access to for the search API
    * ``search_case_sensitive`` (bool or False): whether text related search is case sensitive or not
    * ``rank`` (bool or True): whether to enable the rank API or not
    * ``rank_data_format`` (str or 'records'): same as ``data_format`` but for the rank API only
    * ``rank_url`` (str or '/api/rank'): the rank API url path

Authors
-------
Richard Wen <rrwen.dev@gmail.com>, Siyuan Liu <siyuan.liu@ryerson.ca>

Example
-------
Add the following to ``config.toml`` to include this plugin:

.. code::

    [[plugin]]

        name = 'wits_hiresmart_api'

        [plugin.parameters]
        host = 'localhost'
        port = 8080
        data_format = 'records'
        words_model = './models/word_vector'
        upload_url = '/api/upload'
        search_url = 'api/search'
        rank_url = '/api/rank'
        similar_words_url = '/api/similar'
        upload_table = 'resumes'
        upload_text_column = 'text'
        upload_folder = './uploads'
        search_restricted_columns = ['file_path']

Alter ``host`` and ``port`` to suit the required host and port to serve the application at.
"""

from flask import send_from_directory
from flask_restful import Api, abort, Resource, reqparse
from flask_restful import request as req
from pandas.api.types import is_numeric_dtype
from pathlib import Path
from werkzeug.utils import secure_filename
from wits_resume_data.file import file_to_text
from wits_resume_ranker import calculate_scores, score_functions
from wits_resume_model.word import WordVectorModel

import datetime
import os
import pandas as pd
import time
import uuid
import werkzeug

# (init_settings) Settings for plugin
name = 'wits_hiresmart_api'
kind = 'WebApplicationPlugin'
parameters = dict(
    copyright = '© 2020 WITS Consulting Inc.',
    host = 'localhost',
    port = 8080,
    data_format = 'records',
    words_model = './models/word_vector',
    upload = True,
    upload_url = '/api/upload',
    upload_folder = './uploads',
    upload_table = 'resumes',
    upload_date_column = 'upload_date',
    upload_text_column = 'text',
    upload_file_column = 'file_path',
    upload_original_file_column = 'original_file_name',
    download = True,
    download_url = '/api/download',
    download_column = 'download_link',
    similar_words = True,
    similar_words_url = '/api/similar',
    search = True,
    search_url = '/api/search',
    search_limit = None,
    search_columns = None,
    search_restricted_columns = [
        'file_path'
    ],
    search_case_sensitive = False,
    rank = True,
    rank_url = '/api/rank'
)
for k in ['similar_words_data_format', 'search_data_format', 'rank_data_format']:
    parameters[k] = parameters['data_format']

# (init_packages) Plugin packages
required_packages = [
    'flask_restful',
    'pandas',
    'pip:git+https://github.com/ryerson-ggl/wits_resume_data.git',
    'pip:git+https://github.com/ryerson-ggl/wits_resume_model.git',
    'pip:git+https://github.com/ryerson-ggl/wits_resume_ranker.git',
    'werkzeug'
]

# (init_loop) Plugin loop behaviour
loop = False
run_separately = False

# (init_operator) Known list of operators
__operators__ = {
    'eq': '=',
    'gt': '>',
    'gte': '>=',
    'ne': '!=',
    'lt': '<',
    'lte': '<=',
    'like': 'LIKE',
    'similar': 'SIMILAR TO'
}

# (init_parser) Create request parser
__parser__ = reqparse.RequestParser()
__parser__.add_argument('file[]', type = werkzeug.datastructures.FileStorage, action = 'append', location = 'files', required = True)

# (init_errors) Error returns for API
__errors__ = {
    'InternalServerError': {
        'status': 500,
        'status_type': 'ERROR'
    },
    'BadRequest': {
        'status': 400,
        'status_type': 'ERROR'
    },
    'NotFound': {
        'status': 404,
        'status_type': 'ERROR'
    }
}

# (init_run) Function to run on every loop
def run(database, app, logger, **kwargs):
    print = logger.info

    # (init_run_vars) Initial variables
    out = app
    api = Api(app, errors = __errors__, catch_all_404s = True)

    # (init_run_kwargs) Add keys to kwargs
    kwargs['database'] = database
    kwargs['print'] = print

    # (init_run_resource) Set certain resources on or off
    upload = kwargs.get('upload', parameters['upload'])
    download = kwargs.get('download', parameters['download'])
    similar_words = kwargs.get('similar_words', parameters['similar_words'])
    search = kwargs.get('search', parameters['search'])
    rank = kwargs.get('rank', parameters['rank'])

    # (init_run_upload) Add resume upload via POST
    if upload:

        # (init_run_upload_folder) Setup folder and vars for post resource
        upload_folder = kwargs.get('upload_folder', './uploads')
        Path(upload_folder).mkdir(parents = True, exist_ok = True)

        # (init_run_upload_add) Add upload resource
        upload_url = kwargs.get('upload_url', parameters['upload_url'])
        api.add_resource(UploadResume, upload_url, resource_class_kwargs = kwargs)
        print('Listening wits_hiresmart_api at {}...'.format(upload_url))

    # (init_run_download) Add resume download via GET
    if download:
        download_url = kwargs.get('download_url', parameters['download_url'])
        api.add_resource(DownloadResume, download_url + '/<file_name>', resource_class_kwargs = kwargs)
        print('Listening wits_hiresmart_api at {}...'.format(download_url))

    # (init_run_word) Add resume word similarity via GET
    if similar_words:
        similar_words_url = kwargs.get('similar_words_url', parameters['similar_words_url'])
        api.add_resource(SimilarWords, similar_words_url, resource_class_kwargs = kwargs)
        print('Listening wits_hiresmart_api at {}...'.format(similar_words_url))

    # (init_run_search) Add search resumes via GET
    if search:
        search_url = kwargs.get('search_url', parameters['search_url'])
        api.add_resource(SearchResume, search_url, resource_class_kwargs = kwargs)
        print('Listening wits_hiresmart_api at {}...'.format(search_url))

    # (init_run_rank) Add ranker via GET
    if rank:
        rank_url = kwargs.get('rank_url', parameters['rank_url'])
        api.add_resource(RankResume, rank_url, resource_class_kwargs = kwargs)
        print('Listening wits_hiresmart_api at {}...'.format(rank_url))
    return out

# (DownloadResume) Resource for downloading resumes
class DownloadResume(Resource):

    # (DownloadResume_init) Initial parameters for resource
    def __init__(self, **kwargs):
        self.print = kwargs.get('print', print)
        self.download_url = kwargs.get('download_url', parameters['download_url'])
        self.upload_folder = kwargs.get('upload_folder', parameters['upload_folder'])
    
    # (DownloadResume_get) GET request for resume downloads
    def get(self, file_name):
        print = self.print

        # (DownloadResume_get_rank_vars) Download variables
        download_url = self.download_url
        upload_folder = self.upload_folder

        # (DownloadResume_get_request) Get request fields
        ip = req.remote_addr

        # (DownloadResume_get_download) Send the file as a response
        out = send_from_directory(directory = upload_folder, filename = file_name)
        print('GET {}/{} ({}) from {}'.format(download_url, file_name, 200, ip))
        return out

# (RankResume) Resource for ranking resumes
class RankResume(Resource):

    # (RankResume_init) Initial parameters for resource
    def __init__(self, **kwargs):
        self.print = kwargs.get('print', print)
        self.copyright = kwargs.get('copyright', parameters['copyright'])
        self.data_format = kwargs.get('data_format', parameters['data_format'])
        self.rank_url = kwargs.get('rank_url', parameters['rank_url'])
        self.rank_data_format = kwargs.get('rank_data_format', parameters['rank_data_format'])
        self.words_model = kwargs.get('words_model', parameters['words_model'])
        self.upload_text_column = kwargs.get('upload_text_column', parameters['upload_text_column'])
        self.kwargs = kwargs

    # (RankResume_get) GET request for ranking resumes
    def get(self):

        # (RankResume_get_vars) Initial variables
        print = self.print
        copyright = self.copyright

        # (RankResume_get_rank_vars) Rank variables
        rank_url = self.rank_url
        rank_text_column = self.upload_text_column
        data_format = self.rank_data_format

        # (RankResume_get_request) Get request fields
        query = req.args.to_dict()
        top = int(query.pop('top', 10))
        ip = req.remote_addr

        # (RankResume_get_response) Format initial response format
        out = {'copyright': copyright}

        # (RankResume_get_query) Get ranking query keywords
        rank_query = {}
        rank_query_keys = [k + '_points' for k in score_functions]
        for k in rank_query_keys:
            if k in query:
                rank_query[k] = float(query.pop(k))
        rank_query['keywords'] = query.pop('keywords').split(',')

        # (RankResume_get_search) Search resumes first to filter potential resumes
        query[rank_text_column + '_similar'] = '|'.join('%{}%'.format(word) for word in rank_query['keywords'])
        search_data = SearchResume(**self.kwargs).get(query = query)[0]['data']
        search_data = pd.DataFrame(search_data)

        # (RankResume_get_model) Load the word vector model to use
        model_path = self.words_model
        file_name = os.path.basename(model_path)
        folder_path = os.path.dirname(model_path)
        model = WordVectorModel(file_name = file_name, folder_path = folder_path)
        
        # (RankResume_get_resumes) Get ranked resumes
        rank_query['text'] = search_data[rank_text_column]
        rank_query['word_vector_model'] = model
        rank_data = calculate_scores(**rank_query)

        # (RankResume_get_format) Format ranked resumes and return only top candidates
        data = pd.concat([search_data, rank_data], axis = 1)
        data = data.sort_values('rank').head(top)
        records = len(data)
        data = data.to_dict(data_format)

        # (RankResume_get_return) Print messages and respond with data
        out.update({
            'status': 200,
            'status_type': 'OK',
            'message': '{} records returned!'.format(records),
            'data': data
        })
        print('GET {} ({}) from {}'.format(rank_url, 200, ip))
        return out, 200

# (SearchResume) Resource for searching resumes
class SearchResume(Resource):

    # (SearchResume_init) Initial parameters for resource
    def __init__(self, **kwargs):

        # (SearchResume_init_attr) Standard attributes
        self.database = kwargs.get('database')
        self.print = kwargs.get('print', print)
        self.copyright = kwargs.get('copyright', parameters['copyright'])
        self.data_format = kwargs.get('data_format', parameters['data_format'])
        self.upload_file_column = kwargs.get('upload_file_column', parameters['upload_file_column'])
        self.download = kwargs.get('download', parameters['download'])
        self.download_url = kwargs.get('download_url', parameters['download_url'])
        self.download_column = kwargs.get('download_column', parameters['download_column'])
        self.search_url = kwargs.get('search_url', parameters['search_url'])
        self.search_limit = kwargs.get('search_limit', parameters['search_limit'])
        self.search_table = kwargs.get('upload_table', parameters['upload_table'])
        self.search_columns = kwargs.get('search_columns', parameters['search_columns'])
        self.search_restricted_columns = kwargs.get('search_restricted_columns', parameters['search_restricted_columns'])
        self.search_case_sensitive = kwargs.get('search_case_sensitive', parameters['search_case_sensitive'])
        self.search_data_format = kwargs.get('search_data_format', parameters['search_data_format'])
        
        # (SearchResume_init_column) Dynamic column attributes
        search_sample = self.database.read(self.search_table, limit = 1)
        self.search_column_types = [search_sample[c] for c in search_sample.columns]
        self.search_column_types = ['NUMERIC' if is_numeric_dtype(c) else 'VARCHAR' for c in self.search_column_types]
        self.search_columns = search_sample.columns.to_list() if self.search_columns is None else self.search_columns
    
    # (SearchResume_get) GET request for searching resumes
    def get(self, query = None):
        
        # (SearchResume_get_vars) Initial variables
        database = self.database
        print = self.print
        copyright = self.copyright

        # (SearchResume_get_search_vars) Search variables
        upload_file_column = self.upload_file_column
        download = self.download
        download_url = self.download_url
        download_column = self.download_column
        search_url = self.search_url
        search_table = self.search_table
        search_columns = self.search_columns
        search_column_types = self.search_column_types
        search_restricted_columns = self.search_restricted_columns
        data_format = self.search_data_format

        # (SearchResume_get_request) Get fields from request
        query = req.args.to_dict() if query is None else query
        ip = req.remote_addr
        limit = query.pop('limit') if 'limit' in query else self.search_limit
        case_sensitive = bool(query.pop('case_sensitive')) if 'case_sensitive' in query else self.search_case_sensitive
        only = query.pop('only').split(',') if 'only' in query else None

        # (SearchResume_get_response) Format initial response format
        out = {'copyright': copyright}

        # (SearchResume_get_columns) Get verified search columns
        columns = ['_'.join(k.split('_')[:-1]) if k.split('_')[-1] in __operators__ else k for k in query]
        search_columns = [c for c in search_columns if c not in search_restricted_columns] if search_restricted_columns is not None else search_columns
        columns_allowed = all(c in search_columns for c in columns)
        columns_allowed = all(c in search_columns for c in only) and columns_allowed if only is not None else columns_allowed
        
        # (SearchResume_get_restrict) Restrict unverified search columns
        if not columns_allowed:
            status = 400
            print('GET {} ({}) from {}'.format(search_url, status, ip))
            abort(status)

        # (SearchResume_get_params) Get search parameters from query
        operators = [k.split('_')[-1] for k in query]
        operators = [__operators__[k] if k in __operators__ else '=' for k in operators]
        values = list(query.values())
        values = ["'{}'".format(v) if not v.isnumeric() else v for v in values]
        where = ['LOWER({}) {} LOWER({})'.format(c, o, v) if t == 'VARCHAR' and not case_sensitive else '{} {} {}'.format(c, o, v) for c, t, o, v in zip(columns, search_column_types, operators, values)]

        # (SearchResume_get_resumes) Get ranked resumes data
        select = only if only is not None else search_columns
        select = select + [upload_file_column] if download else select
        data = database.read(search_table, select = select, where = where, limit = limit)

        # (SearchResume_get_download) Add download links if download allowed
        if download:
            data[download_column] = data[upload_file_column].map(lambda file_path: '{}/{}'.format(download_url, os.path.basename(file_path)))
            data = data.drop(columns = upload_file_column)

        # (SearchResume_get_return) Return the search response
        data = data.to_dict(data_format)
        status = 200
        out.update({
            'status': status,
            'status_type': 'OK',
            'message': '{} records returned!'.format(len(data)),
            'data': data
        })
        print('GET {} ({}) from {}'.format(search_url, status, ip))
        return out, status

# (SimilarWords) Resource for getting similar words
class SimilarWords(Resource):

    # (SimilarWords_init) Initial parameters for resource
    def __init__(self, **kwargs):
        self.print = kwargs.get('print', print)
        self.copyright = kwargs.get('copyright', parameters['copyright'])
        self.data_format = kwargs.get('data_format', parameters['data_format'])
        self.similar_words_url = kwargs.get('similar_words_url', parameters['similar_words_url'])
        self.words_model = kwargs.get('words_model', parameters['words_model'])
        self.similar_words_data_format = kwargs.get('similar_words_data_format', parameters['similar_words_data_format'])

    # (SimilarWords_get) GET request for similar words
    def get(self):

        # (SimilarWords_get_vars) Initial variables
        print = self.print
        copyright = self.copyright

        # (SimilarWords_get_vars_words) Initial similar words variables
        similar_words_url = self.similar_words_url
        model_path = self.words_model
        data_format = self.similar_words_data_format

        # (SimilarWords_get_request) Get request fields
        query = req.args
        ip = req.remote_addr

        # (SimilarWords_get_response) Format initial response format
        out = {'copyright': copyright}
        
        # (SimilarWords_get_error) Return error if required query params are not met
        query_params = ['keywords']
        params_exist = all(p in query for p in query_params)
        if not params_exist:
            status = 400
            print('GET {} ({}) from {}'.format(similar_words_url, status, ip))
            abort(status)

        # (SimilarWords_get_similar) Get word similarity if model exists and params exist
        model_exists = os.path.isfile(model_path + '.model')
        if model_exists:

            # (SimilarWords_get_similar_model) Load the model
            file_name = os.path.basename(model_path)
            folder_path = os.path.dirname(model_path)
            model = WordVectorModel(file_name = file_name, folder_path = folder_path)

            # (SimilarWords_get_similar_params) Get model parameters for similar words from request
            keywords = query.get('keywords').split(',')
            top = int(query.get('top', 10))

            # (SimilarWords_get_similar_words) Get similar words and similarities
            data = model.get_similar_words(keywords = keywords, top = top).to_dict(data_format)

            # (SimilarWords_get_similar_return) Respond with similar words and similarity
            status = 200
            out.update({
                'status': status,
                'status_type': 'OK',
                'message': '{} records returned!'.format(len(data)),
                'data': data
            })
            print('GET {} ({}) from {}'.format(similar_words_url, status, ip))
            return out, status
        
        # (SimilarWords_get_notready) Return not available if model does not exist
        if not model_exists:
            status = 202
            out.update({
                'status': status,
                'status_type': 'UNAVAILABLE',
                'message': 'Resource is not available yet, please try again another time!'
            })
            print('GET {} ({}) from {}'.format(similar_words_url, status, ip))
            return out, status

# (UploadResume) Resource for uploading resumes
class UploadResume(Resource):

    # (UploadResume_init) Initial parameters for resource
    def __init__(self, **kwargs):
        self.database = kwargs.get('database')
        self.print = kwargs.get('print', print)
        self.copyright = kwargs.get('copyright', parameters['copyright'])
        self.upload_url = kwargs.get('upload_url', parameters['upload_url'])
        self.upload_folder = kwargs.get('upload_folder', parameters['upload_folder'])
        self.upload_table = kwargs.get('upload_table', parameters['upload_table'])
        self.upload_date_column = kwargs.get('upload_date_column', parameters['upload_date_column'])
        self.upload_text_column = kwargs.get('upload_text_column', parameters['upload_text_column'])
        self.upload_file_column = kwargs.get('upload_file_column', parameters['upload_file_column'])
        self.upload_original_file_column = kwargs.get('upload_original_file_column', parameters['upload_original_file_column'])

    # (UploadResume_post) POST request for resume uploads
    def post(self):

        # (UploadResume_post_vars) Initial variables
        database = self.database
        print = self.print
        copyright = self.copyright

        # (UploadResume_post_vars_upload) Initial upload variables
        upload_url = self.upload_url
        upload_folder = self.upload_folder
        upload_table = self.upload_table
        upload_date_column = self.upload_date_column
        upload_text_column = self.upload_text_column
        upload_file_column = self.upload_file_column
        upload_original_file_column = self.upload_original_file_column

        # (UploadResume_get_request) Get request fields
        ip = req.remote_addr

        # (UploadResume_get_response) Format initial response format
        out = {'copyright': copyright}

        # (UploadResume_post_upload) Upload the file if it exists or return error if not
        resumes = __parser__.parse_args()['file[]']
        for resume in resumes:

            # (UploadResume_post_upload_file) Get the original file name and extension
            original_file_name = resume.filename
            ext = os.path.splitext(original_file_name)[1]

            # (UploadResume_post_upload_save) Save the file in the upload folder with a unique file name
            file_name = secure_filename('{}{}'.format(uuid.uuid4().hex, ext))
            file_path = os.path.join(upload_folder, file_name)
            resume.save(file_path)

            # (UploadResume_post_upload_date) Get ISO 8601 timestamp from local timezone
            utc_offset_sec = time.altzone if time.localtime().tm_isdst else time.timezone
            utc_offset = datetime.timedelta(seconds=-utc_offset_sec)
            timestamp = datetime.datetime.now().replace(tzinfo=datetime.timezone(offset=utc_offset)).isoformat()

            # (UploadResume_post_upload_data) Create file path and read text dataframe
            file_data = pd.DataFrame({
                upload_date_column: [timestamp],
                upload_text_column: [file_to_text(file_path)],
                upload_file_column: [os.path.abspath(file_path)],
                upload_original_file_column : [original_file_name]
            })

            # (UploadResume_post_upload_database) Upload dataframe to database
            if database.has_table(upload_table):
                database.add_columns(file_data.columns.to_list(), upload_table, column_types = 'VARCHAR')
            database.write(file_data, upload_table, if_exists = 'append')
        
        # (UploadResume_post_response) Return the response
        status = 200
        out.update({
            'status': status,
            'status_type': 'OK',
            'message': '{} resume files uploaded!'.format(len(resumes))
        })
        print('POST {} ({}) {} uploaded from {}'.format(upload_url, status, len(resumes), ip))
        return out, status
