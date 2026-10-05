from scripts.check_release import findings, forbidden


def test_audit_redacts_private_key_match():
    content = b'-----BEGIN' + b' PRIVATE KEY-----\nfictional\n-----END PRIVATE KEY-----'
    report = findings(content, 'fixture.txt', 'test')
    assert report == [{'scope': 'test', 'path': 'fixture.txt', 'line': 1,
                       'type': 'private_key_block', 'value': '[REDACTED]'}]
    assert 'fictional' not in str(report)


def test_audit_excludes_private_artifacts_but_allows_synthetic_json_and_examples():
    assert forbidden('data/model.joblib')
    assert forbidden('src/_compiled/recorder.pyc')
    assert forbidden('.env')
    assert not forbidden('.env.example')
    assert not forbidden('examples/synthetic_training_sample.json')
