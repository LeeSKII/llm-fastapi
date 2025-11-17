from fastapi import APIRouter

from ...database.mysql_pool import mysql_pool

router = APIRouter()

@router.get("/", tags=["yus"])
async def test():
    return {"message": "Hello, yus!"}


@router.get("/user/{user_no}", tags=["yus"])
async def get_user(user_no: int):
    select_user_query = "SELECT * FROM h_org_user WHERE employeeNo = %s"
    user = await mysql_pool.fetch_one(select_user_query, user_no)
    return {"message": f"Hello, {user_no}!", "user": user}