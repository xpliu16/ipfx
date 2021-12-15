import os
import glob
import logging
import pg8000
from pathlib import Path
from allensdk.core.authentication import credential_injector
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP

from ipfx.string_utils import to_str


TIMEOUT = os.environ.get(
    "IPFX_LIMS_TIMEOUT",
    os.environ.get(
        "IPFX_TEST_TIMEOUT",
        None
    )
)
if TIMEOUT is not None:
    TIMEOUT = float(TIMEOUT)  # type: ignore


@credential_injector(LIMS_DB_CREDENTIAL_MAP)
def _connect(user, host, dbname, password, port, timeout=TIMEOUT):

    conn = pg8000.connect(
        user=user, 
        host=host, 
        database=dbname, 
        password=password, 
        port=int(port),
        timeout=timeout
    )
    return conn, conn.cursor()


def able_to_connect_to_lims():

    try:
        conn, cursor = _connect()
        cursor.close()
        conn.close()
    except pg8000.Error:
        # the connection failed
        return False
    except TypeError:
        # a credential was missing
        return False

    return True


def _select(cursor, query, parameters=None):
    if parameters is None:
        cursor.execute(query)
    else:
        pg8000.paramstyle = 'numeric'
        cursor.execute(query, parameters)
    columns = [ to_str(d[0]) for d in cursor.description ]
    return [ dict(zip(columns, c)) for c in cursor.fetchall() ]


def query(query, parameters=None):
    conn, cursor = _connect()
    try:
        results = _select(cursor, query, parameters=parameters)
    finally:
        cursor.close()
        conn.close()
    return results

def get_sweep_states(specimen_id):

    sweep_states = []

    res = query("""
        select sweep_number, workflow_state from ephys_sweeps
        where specimen_id = %d
        """ % specimen_id)

    for sweep in res:
        # only care about manual calls
        if sweep["workflow_state"] == "manual_passed":
            sweep_states.append({'sweep_number': sweep["sweep_number"],
                                 'passed': True})
        elif sweep["workflow_state"] == "manual_failed":
            sweep_states.append({'sweep_number': sweep["sweep_number"],
                                 'passed': False})

    return sweep_states


def get_stimuli_description():

    stims = query("""
    select ersn.name as stimulus_code, est.name as stimulus_name from ephys_raw_stimulus_names ersn
    join ephys_stimulus_types est on ersn.ephys_stimulus_type_id = est.id
    """)

    return stims


def get_specimen_info_from_lims_by_id(specimen_id):

    result = query("""
                  SELECT s.name, s.ephys_roi_result_id, s.id
                  FROM specimens s
                  WHERE s.id = %s
                  """ % specimen_id)
    if len(result) == 0:
        logging.info("No result from query to find specimen information")
        return None, None, None

    result = result[0]

    if result:
        return result["name"], result["ephys_roi_result_id"], result["id"]
    else:
        logging.info("Could not find specimen {:d}".format(specimen_id))
        return None, None, None


def get_nwb_path_from_lims(specimen_id, ftype='EphysNWB2'):
    """
    Find network path to stored NWB file

    Parameters
    ----------
    specimen_id: int
    ftype: str, defaults to 'EphysNWB2', or try 'NWB' or 'NWBIgor'

    Returns
    -------
    full path of the nwb file

    """

    result = query("""
    SELECT f.filename, f.storage_directory 
    FROM specimens sp
    JOIN ephys_roi_results err ON sp.ephys_roi_result_id = err.id
    JOIN well_known_files f ON f.attachable_id = err.id
    JOIN well_known_file_types ftype ON f.well_known_file_type_id = ftype.id
    WHERE f.attachable_type = 'EphysRoiResult' 
    AND sp.id = %s 
    AND ftype.name = '%s'
    """ % (specimen_id, ftype))
    
    if len(result) == 0:
        logging.info("No NWB file for specimen.")
        return None
    
    result = result[0]

    if result:
        nwb_path = result["storage_directory"] + result["filename"]
        return fix_network_path(nwb_path)
    else:
        logging.info("Cannot find NWB file")
        return None


def get_igorh5_path_from_lims(ephys_roi_result):

    sql = """
    SELECT f.filename, f.storage_directory
    FROM well_known_files f
    WHERE f.attachable_type = 'EphysRoiResult'
    AND f.attachable_id = %s
    AND f.well_known_file_type_id = 306905526
    """ % ephys_roi_result

    result = query(sql)
    if len(result) == 0:
        logging.info("No result from query to find Igor H5 file")
        return None

    result = result[0]

    if result:
        h5_path = result["storage_directory"] + result["filename"]
        return fix_network_path(h5_path)
    else:
        logging.info("Cannot find Igor H5 file")
        return None

def fix_network_path(lims_path):
    # Need to have double slash for network drive
    if not lims_path.startswith('//'):
        lims_path = '/' + lims_path
    return str(Path(lims_path))

def project_specimen_ids(project, passed_only=True):

    SQL = """
        SELECT sp.id FROM specimens sp
        JOIN ephys_roi_results err ON sp.ephys_roi_result_id = err.id
        JOIN projects prj ON prj.id = sp.project_id
        WHERE prj.code = '%s'
        """ % project

    if passed_only:
        SQL += "AND err.workflow_state = 'manual_passed'"

    results = query(SQL)
    sp_ids = [d["id"] for d in results]
    return sp_ids


def get_fx_output_json(specimen_id):
    """
    Find in LIMS the full path to the json output of the feature extraction module
    If more than one file exists, then chose the latest version

    Parameters
    ----------
    specimen_id

    Returns
    -------
    file_name: string
    """
    NO_SPECIMEN = "No_specimen_in_LIMS"
    NO_OUTPUT_FILE = "No_feature_extraction_output"
    
    sql = """
    select err.storage_directory, err.id
    from specimens sp
    join ephys_roi_results err on err.id = sp.ephys_roi_result_id
    where sp.id = %d
    """ % specimen_id

    res = query(sql)
    if res:
        err_dir = res[0]["storage_directory"]

        file_list = glob.glob(fix_network_path(os.path.join(err_dir, '*EPHYS_FEATURE_EXTRACTION_*_output.json')))
        if file_list:
            latest_file = max(file_list, key=os.path.getctime)   # get the most recent file
            return latest_file
        else:
            return NO_OUTPUT_FILE
    else:
        return NO_SPECIMEN

